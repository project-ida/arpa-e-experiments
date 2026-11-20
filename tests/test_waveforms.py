"""Generated ROOT fixtures exercise timing, file boundaries, and orchestration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from libs.waveforms import WaveformError, find_root_files, load_waveforms, read_waveforms


class WaveformTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.start = pd.Timestamp("2025-09-03 12:00:00")
        self.end = pd.Timestamp("2025-09-03 12:00:03")

    def root_file(self, name, timestamps, *, energies=None, short=None, fixed_samples=False):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        count = len(timestamps)
        energies = list(range(10, 10 + count)) if energies is None else energies
        branches = {"Timestamp": "uint64", "Energy": "uint16", "Samples": "3 * uint16" if fixed_samples else "var * uint16"}
        values = {"Timestamp": np.asarray(timestamps, dtype=np.uint64), "Energy": np.asarray(energies, dtype=np.uint16)}
        if count:
            values["Samples"] = np.asarray([[i, i + 1, i + 2] for i in range(count)], dtype=np.uint16) if fixed_samples else ak.Array([[i] * (i + 2) for i in range(count)])
        if short is not None:
            branches["EnergyShort"] = "uint16"
            values["EnergyShort"] = np.asarray(short, dtype=np.uint16)
        with uproot.recreate(path) as root_file:
            tree = root_file.mktree("Data_R", branches)
            if count:
                tree.extend(values)
        return path

    def two_files(self):
        first = self.root_file(
            "20250903_115959-20250903_120001_DataR_CH6@first.root2",
            [10 * 10**12, 11 * 10**12, 12 * 10**12], energies=[10, 20, 30], short=[5, 10, 15],
        )
        second = self.root_file(
            "20250903_120001-20250903_120004_DataR_CH6@second.root",
            [12_500_000_000_000, 13 * 10**12, 15 * 10**12], energies=[40, 50, 60],
        )
        return first, second

    def index_rows(self, paths):
        rows = []
        for path in paths:
            bounds = path.name.split("_DataR")[0].split("-")
            first, last = pd.to_datetime(bounds, format="%Y%m%d_%H%M%S")
            rows.append({"time": last, "computer": "daq", "dir": "run/data", "file": path.name, "device": "caen8ch", "file_start": first, "file_end": last})
        return pd.DataFrame(rows)

    def test_combines_files_and_clips_to_half_open_interval(self):
        first, second = self.two_files()
        result = read_waveforms([second, first], start=self.start, end=self.end, step_size=1)
        expected = pd.to_datetime(["2025-09-03 12:00:00", "2025-09-03 12:00:01", "2025-09-03 12:00:01.500", "2025-09-03 12:00:02"], format="mixed")
        pd.testing.assert_index_equal(result.index, pd.DatetimeIndex(expected, name="time"))
        self.assertEqual(result["Energy"].tolist(), [20, 30, 40, 50])
        self.assertEqual(result["Timestamp"].tolist(), [11 * 10**12, 12 * 10**12, 12_500_000_000_000, 13 * 10**12])
        self.assertEqual(result["Waveform"].iloc[2].tolist(), [0, 0])
        self.assertTrue(result.loc[result["SourceFile"] == str(second), "PSP"].isna().all())
        self.assertEqual(len(result.attrs["root_files"]), 2)

    def test_raw_picoseconds_above_signed_int64_are_preserved(self):
        clock = 10_000_000_000_000_000_000
        path = self.root_file("20250903_120000-20250903_120001_DataR_CH6@precise.root2", [clock, clock + 500, clock + 1500])
        result = read_waveforms(path, step_size=1)
        anchor = pd.Timestamp("2025-09-03 12:00:01")
        self.assertEqual(result["Timestamp"].tolist(), [clock, clock + 500, clock + 1500])
        self.assertEqual(result.index.tolist(), [anchor - pd.Timedelta(2, "ns"), anchor - pd.Timedelta(1, "ns"), anchor])

    def test_end_anchor_can_place_first_pulse_before_filename_start(self):
        # Absolute event fractions were removed from the file's end anchor.
        # This is why discovery needs a margin before filename start as well.
        path = self.root_file(
            "20250903_120003-20250903_120004_DataR_CH6@boundary.root2",
            [13_100_000_000_000, 14_800_000_000_000],
        )
        result = read_waveforms(path, start="2025-09-03 12:00:02.2", end="2025-09-03 12:00:02.4")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index[0], pd.Timestamp("2025-09-03 12:00:02.3"))

    def test_saved_csv_metadata_can_reopen_downloads(self):
        first, second = self.two_files()
        result = read_waveforms([first, second], start=self.start, end=self.end)
        manifest = self.base / "files.csv"
        pd.DataFrame(result.attrs["root_files"]).to_csv(manifest, index=False)
        reopened = read_waveforms(pd.read_csv(manifest), start=self.start, end=self.end)
        pd.testing.assert_frame_equal(result, reopened)

    def test_uses_latest_timestamp_when_entries_are_unsorted(self):
        path = self.root_file("20250903_120000-20250903_120001_DataR_CH6@unsorted.root2", [2000, 0, 1000], energies=[10, 20, 30])
        result = read_waveforms(path)
        self.assertEqual(result["Energy"].tolist(), [20, 30, 10])
        self.assertEqual(result.index[-1], pd.Timestamp("2025-09-03 12:00:01"))

    def test_explicit_metadata_supports_unprefixed_local_files(self):
        path = self.root_file("original.root2", [1000, 2000], fixed_samples=True)
        metadata = pd.DataFrame([{"local_path": str(path), "time": "20250903_120002"}])
        with patch("libs.waveforms.read_sql") as query, patch("libs.waveforms.download_root_file") as download:
            result = read_waveforms(metadata)
        query.assert_not_called()
        download.assert_not_called()
        self.assertEqual(result.index[-1], pd.Timestamp("2025-09-03 12:00:02"))
        np.testing.assert_array_equal(result["Waveform"].iloc[0], [0, 1, 2])
        with self.assertRaisesRegex(WaveformError, "Cannot infer the end time"):
            read_waveforms(path)

    def test_repeated_paths_do_not_duplicate_pulses_but_equal_times_are_kept(self):
        path = self.root_file("20250903_120000-20250903_120001_DataR_CH6@equal.root2", [1000, 1000, 2000])
        result = read_waveforms([path, path])
        self.assertEqual(len(result), 3)
        self.assertTrue(result.index.duplicated().any())

    def test_conflicting_local_anchors_are_rejected(self):
        first, _ = self.two_files()
        metadata = pd.DataFrame([
            {"local_path": str(first), "time": "2025-09-03 12:00:01"},
            {"local_path": str(first), "time": "2025-09-03 12:00:02"},
        ])
        with self.assertRaisesRegex(WaveformError, "Conflicting end times"):
            read_waveforms(metadata)

    def test_missing_file_or_branch_is_not_silently_skipped(self):
        first, _ = self.two_files()
        with self.assertRaises(FileNotFoundError):
            read_waveforms([first, self.base / "missing.root2"])
        bad = self.base / "20250903_120001-20250903_120002_DataR_CH6@bad.root2"
        with uproot.recreate(bad) as root_file:
            root_file.mktree("Data_R", {"Timestamp": "uint64", "Energy": "uint16"})
        with self.assertRaisesRegex(WaveformError, "Samples"):
            read_waveforms([first, bad])

    def test_undefined_psp_and_files_without_short_energy(self):
        path = self.root_file("20250903_120000-20250903_120001_DataR_CH6@zero.root2", [0, 10**12], energies=[0, 100], short=[0, 40])
        result = read_waveforms(path)
        self.assertTrue(pd.isna(result["PSP"].iloc[0]))
        self.assertAlmostEqual(result["PSP"].iloc[1], 0.6)
        _, without_short = self.two_files()
        self.assertNotIn("PSP", read_waveforms(without_short).columns)

    def test_empty_tree_and_empty_interval_have_valid_dataframe_schema(self):
        empty = self.root_file("20250903_120000-20250903_120001_DataR_CH6@empty.root2", [], short=[])
        first, _ = self.two_files()
        for files, start, end in [([empty], None, None), ([first], "2025-09-03 13:00:00", "2025-09-03 13:01:00")]:
            with self.subTest(files=files):
                result = read_waveforms(files, start=start, end=end)
                self.assertTrue(result.empty)
                self.assertIsInstance(result.index, pd.DatetimeIndex)
                self.assertIn("PSP", result)
                self.assertEqual(len(result.attrs["root_files"]), 1)

    def test_invalid_windows_are_rejected(self):
        first, _ = self.two_files()
        for start, end in [(self.end, self.start), (self.start, self.start), ("2025-09-03T12:00:00Z", self.end), ("bad time", self.end)]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                read_waveforms(first, start=start, end=end)

    def test_find_files_passes_connection_and_bound_values_and_removes_duplicate_rows(self):
        paths = self.two_files()
        rows = self.index_rows(paths)
        connection = object()
        device = "caen8ch' OR 1=1 --"
        with patch("libs.waveforms.read_sql", return_value=pd.concat([rows, rows.iloc[:1]])) as query:
            result = find_root_files(connection, device=device, channel=6, start=self.start, end=self.end)
        self.assertEqual(len(result), 2)
        self.assertIs(query.call_args.args[1], connection)
        self.assertNotIn(device, query.call_args.args[0])
        self.assertEqual(query.call_args.kwargs["params"]["device"], device)
        self.assertEqual(query.call_args.kwargs["params"]["file_pattern"], "%CH6@%")
        self.assertEqual(query.call_args.kwargs["params"]["start"], self.start)

    def test_conflicting_database_anchors_are_rejected(self):
        paths = self.two_files()
        rows = self.index_rows(paths)
        duplicate = rows.iloc[:1].copy()
        duplicate["time"] += pd.Timedelta(seconds=1)
        with patch("libs.waveforms.read_sql", return_value=pd.concat([rows, duplicate])):
            with self.assertRaisesRegex(WaveformError, "conflicting end times"):
                find_root_files(object(), device="caen8ch", channel=6, start=self.start, end=self.end)

    def test_load_downloads_each_selected_file_then_returns_interval_pulses(self):
        paths = self.two_files()
        rows = self.index_rows(paths)
        by_name = {path.name: path for path in paths}
        with patch("libs.waveforms.find_root_files", return_value=rows) as find, patch("libs.waveforms.download_root_file", side_effect=lambda row, **kwargs: by_name[row["file"]]) as download:
            result = load_waveforms(object(), device="caen8ch", channel=6, start=self.start, end=self.end, show_progress=False, step_size=1)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(result["Energy"].tolist(), [20, 30, 40, 50])
        self.assertEqual(find.call_args.kwargs["end"], self.end)
        metadata = pd.DataFrame(result.attrs["root_files"])
        offline = read_waveforms(metadata, start=self.start, end=self.end)
        pd.testing.assert_frame_equal(result, offline)

    def test_equal_basenames_use_separate_download_directories(self):
        paths = self.two_files()
        rows = self.index_rows(paths)
        rows.loc[1, "file"] = rows.loc[0, "file"]
        rows.loc[1, "computer"] = "another-computer"
        with patch("libs.waveforms.find_root_files", return_value=rows), patch("libs.waveforms.download_root_file", side_effect=paths) as download:
            load_waveforms(object(), device="caen8ch", channel=6, start=self.start, end=self.end, show_progress=False)
        self.assertNotEqual(download.call_args_list[0].kwargs["download_dir"], download.call_args_list[1].kwargs["download_dir"])

    def test_no_matches_and_failed_download_raise(self):
        with patch("libs.waveforms.find_root_files", return_value=pd.DataFrame()), patch("libs.waveforms.download_root_file") as download:
            with self.assertRaises(FileNotFoundError):
                load_waveforms(object(), device="caen8ch", channel=6, start=self.start, end=self.end, show_progress=False)
            download.assert_not_called()
        rows = self.index_rows(self.two_files())
        with patch("libs.waveforms.find_root_files", return_value=rows), patch("libs.waveforms.download_root_file", side_effect=RuntimeError("download failed")):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                load_waveforms(object(), device="caen8ch", channel=6, start=self.start, end=self.end, show_progress=False)


if __name__ == "__main__":
    unittest.main()
