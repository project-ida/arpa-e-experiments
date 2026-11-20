"""Client transfer checks; no live Nucleonics server is required."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from libs.root_files_api import ROOTFileAPIError, download_root_file


class RootFilesAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.download_dir = Path(self.directory.name)
        self.row = {"computer": "acquisition", "dir": "run folder\\data", "file": "CH4@test.root"}

    def response(self, chunks, *, status=200, headers=None):
        response = Mock()
        response.status_code = status
        response.headers = headers or {}
        response.iter_content.return_value = iter(chunks)
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        return context

    def test_downloads_bytes_with_progress_and_exact_index_fields(self):
        payload = b"root\x00\x01waveform-data"
        events = []
        response = self.response([b"r", b"", b"oot", payload[4:]], headers={"Content-Length": str(len(payload))})
        with patch("libs.root_files_api.requests.get", return_value=response) as get:
            filename = download_root_file(self.row, self.download_dir, progress=lambda *args: events.append(args))
        self.assertEqual(filename.read_bytes(), payload)
        self.assertEqual(events[0], (0, len(payload)))
        self.assertEqual(events[-1], (len(payload), len(payload)))
        self.assertEqual(get.call_args.kwargs["params"], self.row)
        self.assertTrue(get.call_args.kwargs["stream"])
        self.assertEqual(list(self.download_dir.glob("*.part")), [])

    def test_download_without_length_and_with_custom_base_url(self):
        response = self.response([b"root-data"])
        with patch("libs.root_files_api.requests.get", return_value=response) as get:
            filename = download_root_file(self.row, self.download_dir, base_url="https://example.org/api/")
        self.assertEqual(filename.read_bytes(), b"root-data")
        self.assertEqual(get.call_args.args[0], "https://example.org/api/root-files/download")

    def test_downloads_root2_files_without_changing_filename(self):
        for extension in (".root2", ".ROOT2"):
            row = dict(self.row, file="CH4@test" + extension)
            with self.subTest(extension=extension):
                with patch("libs.root_files_api.requests.get", return_value=self.response([b"root-data"])) as get:
                    filename = download_root_file(row, self.download_dir)
                self.assertEqual(filename.name, row["file"])
                self.assertEqual(filename.read_bytes(), b"root-data")
                self.assertEqual(get.call_args.kwargs["params"], row)

    def test_replaces_existing_file_only_after_success(self):
        destination = self.download_dir / self.row["file"]
        destination.write_bytes(b"old ROOT download")
        with patch("libs.root_files_api.requests.get", return_value=self.response([b"root-new"])):
            download_root_file(self.row, self.download_dir)
        self.assertEqual(destination.read_bytes(), b"root-new")

    def test_http_error_exposes_status_and_keeps_existing_file(self):
        destination = self.download_dir / self.row["file"]
        destination.write_bytes(b"existing")
        with patch("libs.root_files_api.requests.get", return_value=self.response([], status=404)):
            with self.assertRaises(ROOTFileAPIError) as error:
                download_root_file(self.row, self.download_dir)
        self.assertEqual(error.exception.status_code, 404)
        self.assertEqual(str(error.exception), "ROOT download failed (HTTP 404).")
        self.assertEqual(destination.read_bytes(), b"existing")

    def test_html_and_empty_responses_are_rejected(self):
        for chunks in ([b"<html>login page</html>"], []):
            with self.subTest(chunks=chunks):
                with patch("libs.root_files_api.requests.get", return_value=self.response(chunks)):
                    with self.assertRaises(ROOTFileAPIError):
                        download_root_file(self.row, self.download_dir)
                self.assertEqual(list(self.download_dir.iterdir()), [])

    def test_short_transfer_removes_partial_and_preserves_previous_download(self):
        destination = self.download_dir / self.row["file"]
        destination.write_bytes(b"existing")
        response = self.response([b"root-short"], headers={"Content-Length": "100"})
        with patch("libs.root_files_api.requests.get", return_value=response):
            with self.assertRaisesRegex(ROOTFileAPIError, "Incomplete ROOT download"):
                download_root_file(self.row, self.download_dir)
        self.assertEqual(destination.read_bytes(), b"existing")
        self.assertEqual(list(self.download_dir.glob("*.part")), [])

    def test_interrupted_stream_removes_partial(self):
        def interrupted():
            yield b"root-partial"
            raise requests.exceptions.ChunkedEncodingError("connection closed")

        response = self.response(interrupted())
        with patch("libs.root_files_api.requests.get", return_value=response):
            with self.assertRaisesRegex(ROOTFileAPIError, "connection closed"):
                download_root_file(self.row, self.download_dir)
        self.assertEqual(list(self.download_dir.iterdir()), [])

    def test_invalid_arguments_are_rejected_before_request(self):
        invalid_rows = (
            {},
            dict(self.row, computer=None),
            dict(self.row, file="../escape.root"),
            dict(self.row, file="../escape.root2"),
            dict(self.row, file="text.txt"),
            dict(self.row, file="test.root3"),
            dict(self.row, file="test.root2.txt"),
        )
        with patch("libs.root_files_api.requests.get") as get:
            for row in invalid_rows:
                with self.subTest(row=row), self.assertRaises(ValueError):
                    download_root_file(row, self.download_dir)
            for url in ("ftp://example.org", "https://example.org?query=1", "https://user:password@example.org"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    download_root_file(self.row, self.download_dir, base_url=url)
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
