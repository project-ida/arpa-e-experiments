"""Download indexed ROOT files through the existing waveform viewer.

Install requests, then pass a row from the root_files SQL table::

    from libs.root_files_api import download_root_file

    filename = download_root_file(closest_root_file)
    with uproot.open(filename) as root_file:
        tree = root_file["Data_R"]

The waveform server must include the /root-files/download route. The default
URL is the existing Nucleonics waveform viewer; base_url can override it.
Downloads are streamed to a temporary file and replace the destination only
after the ROOT header and transfer length have been checked. Each call fetches
the file again, so an old or interrupted download is never reused as a cache.
No SQL credentials are sent to the waveform server.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import requests

__all__ = ["ROOTFileAPIError", "download_root_file"]

DEFAULT_BASE_URL = "http://nucleonics.mit.edu:8080"


class ROOTFileAPIError(RuntimeError):
    """An HTTP error, invalid ROOT response, or incomplete download."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def download_root_file(
    root_file,
    download_dir="downloads/root_files",
    *,
    base_url=DEFAULT_BASE_URL,
    timeout=(10, 600),
    progress=None,
):
    """Download a root_files row and return its local pathlib.Path.

    root_file must provide computer, dir, and file, as a dictionary or pandas
    Series. Both .root and .root2 filenames are accepted.
    progress, when supplied, is called with (bytes_received,
    total_bytes); total_bytes is None if the server does not supply a length.
    timeout is the requests connect/read timeout in seconds.

    An existing destination is replaced only after a successful download.
    Files with the same basename share a destination, but are always fetched
    afresh. HTTP and transfer errors raise ROOTFileAPIError; invalid arguments
    raise ValueError and local filesystem errors raise OSError.
    """
    params = {}
    for key in ("computer", "dir", "file"):
        try:
            value = root_file[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"root_file must contain {key!r}.") from exc
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"root_file[{key!r}] must be a non-empty string.")
        params[key] = value

    filename = params["file"]
    if "/" in filename or "\\" in filename or not filename.lower().endswith((".root", ".root2")):
        raise ValueError("root_file['file'] must be a filename ending in .root or .root2.")

    url = urlsplit(base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.query
        or url.fragment
        or url.username is not None
        or url.password is not None
    ):
        raise ValueError("base_url must be an HTTP(S) server URL without credentials, a query, or a fragment.")

    destination = Path(download_dir) / filename
    temporary = None
    try:
        with requests.get(
            base_url.rstrip("/") + "/root-files/download",
            params=params,
            stream=True,
            timeout=timeout,
            headers={"Accept-Encoding": "identity"},
        ) as response:
            if response.status_code != 200:
                raise ROOTFileAPIError(
                    f"ROOT download failed (HTTP {response.status_code}).",
                    status_code=response.status_code,
                )

            # Content-Length describes the encoded body when compression is used.
            total = None
            if response.headers.get("Content-Encoding", "identity").lower() == "identity":
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        total = int(length)
                    except ValueError as exc:
                        raise ROOTFileAPIError("The server returned an invalid Content-Length.") from exc
                    if total < 0:
                        raise ROOTFileAPIError("The server returned an invalid Content-Length.")

            destination.parent.mkdir(parents=True, exist_ok=True)
            received = 0
            header = b""
            if progress is not None:
                progress(0, total)

            with NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".root-download-", suffix=".part", delete=False
            ) as output:
                temporary = Path(output.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    if len(header) < 4:
                        header = (header + chunk)[:4]
                        if len(header) == 4 and header != b"root":
                            raise ROOTFileAPIError("The server response is not a ROOT file (it may be an HTML error or login page).")
                    output.write(chunk)
                    received += len(chunk)
                    if progress is not None:
                        progress(received, total)

            if header != b"root":
                raise ROOTFileAPIError("The server response is empty or is not a ROOT file.")
            if total is not None and received != total:
                raise ROOTFileAPIError(f"Incomplete ROOT download: received {received} of {total} bytes.")

            temporary.replace(destination)
            return destination
    except requests.RequestException as exc:
        raise ROOTFileAPIError(f"ROOT download failed: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
