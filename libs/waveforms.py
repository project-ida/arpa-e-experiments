"""Find, download, and read CAEN waveforms into a time-indexed DataFrame.

Typical notebook use (connection is an existing sql_api.SQLConnection)::

    df_pulses = load_waveforms(
        connection, device="caen8ch", channel=6,
        start="2025-09-03 11:55:00", end="2025-09-03 12:05:00",
    )

For files already downloaded, no database connection is needed::

    df_pulses = read_waveforms(
        ["20250903_115839-20250903_120201_DataR_CH6@test.root2"],
        start="2025-09-03 12:00:00", end="2025-09-03 12:01:00",
    )

Timing conventions
------------------
These conventions come from ida-devices/caen-rootprocessing.py and
caen-rootpostprocessing.py; they are shared by discovery and local reading.

* ROOT Timestamp is an integer number of picoseconds since acquisition start,
  not since the beginning of each file and not a calendar timestamp. The
  processing scripts estimate acquisition start from the settings file's
  modification time and add Timestamp to obtain absolute event times.
* Processed filenames begin YYYYMMDD_HHMMSS-YYYYMMDD_HHMMSS_. Those are the
  earliest and latest pulse times in that file, with fractional seconds
  discarded. Both .root and .root2 are supported.
* root_files.time is file metadata: the same formatted LAST-pulse time used
  in the filename. It is distinct from the individual pulse timestamps in
  channel tables such as caen8ch_ch6.time (and their subsecond ps field).
* As in the original starter notebook, anchor the latest ROOT pulse to that
  file's root_files.time and work backwards using relative ROOT timestamps.
  For offline reading, use the filename's end time, which is the same anchor.
  Each file has its own anchor; timestamps are never continued from the end
  of a previous file. This also handles unsorted ROOT entries by using max().
* Absolute times inherit the anchor's precision. The usual whole-second
  anchor loses the last pulse's fractional second, shifting all pulses in
  that file earlier by that unknown fraction. Different files can have
  different shifts of less than one second. This does NOT reproduce the
  original channel-table timestamps at subsecond precision. Exact absolute
  alignment needs a more precise anchor or the acquisition-start metadata.
* The pandas index has nanosecond resolution; it is rounded down when ROOT
  offsets contain picoseconds below a nanosecond. The unmodified Timestamp
  column retains the original picosecond values for precise relative work.
* All calendar times use the same naive local clock as the file index and
  processing scripts. Timezone-aware arguments are rejected, not converted.
  Requested windows are half-open: start <= pulse time < end.

Discovery uses the filename overlap approach in ida-scripts/app_plot_waveforms.py,
with a one-second margin on BOTH sides. The margin accounts for discarded
fractions, including the end-anchor convention moving a reconstructed first
pulse slightly before its filename start. Final pulse selection uses the
requested window after reconstruction. No energy/PSP gates or pulse caps are
applied here. Failed downloads/reads raise rather than silently omit files.

Dependencies: pandas, numpy, uproot, and requests. The SQL credentials stay on
the supplied connection, which this module neither creates nor closes.
"""

import hashlib
import numbers
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from .root_files_api import DEFAULT_BASE_URL, download_root_file
from .sql_api import read_sql

__all__ = ["WaveformError", "find_root_files", "load_waveforms", "read_waveforms"]

_FILENAME_TIMES = re.compile(r"^(\d{8}_\d{6})-(\d{8}_\d{6})_")
_FILE_COLUMNS = ["time", "computer", "dir", "file", "device", "file_start", "file_end"]

# Values are bound by sql_api, never interpolated into SQL. Keep the index
# table fixed, and filter by BOTH device and channel: multiple devices can
# have a CH6 file. No LIMIT is used, so a long period is not quietly truncated.
_FIND_FILES_SQL = r"""
WITH indexed_files AS (
    SELECT time, computer, dir, file, device,
           regexp_match(file, '^([0-9]{8}_[0-9]{6})-([0-9]{8}_[0-9]{6})_') AS parts
    FROM root_files
    WHERE lower(device) = lower(%(device)s)
      AND file LIKE %(file_pattern)s
      AND lower(file) ~ '\.root2?$'
), ranges AS (
    SELECT time, computer, dir, file, device,
           to_timestamp(parts[1], 'YYYYMMDD_HH24MISS')::timestamp AS file_start,
           to_timestamp(parts[2], 'YYYYMMDD_HH24MISS')::timestamp AS file_end
    FROM indexed_files
    WHERE parts IS NOT NULL
)
SELECT time, computer, dir, file, device, file_start, file_end
FROM ranges
WHERE file_start - interval '1 second' < %(end)s
  AND file_end + interval '1 second' > %(start)s
ORDER BY file_start, time, computer, dir, file
"""


class WaveformError(RuntimeError):
    """A file or its timing metadata could not be read consistently."""


def _as_time(value, name):
    try:
        if isinstance(value, numbers.Number):
            raise ValueError
        if isinstance(value, str) and re.fullmatch(r"\d{8}_\d{6}", value):
            value = datetime.strptime(value, "%Y%m%d_%H%M%S")
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is not None:
            raise ValueError
        # The output index uses nanoseconds; reject out-of-range values early.
        result.value
        return result
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a valid timezone-naive calendar time.") from exc


def _window(start, end, *, required=False):
    if required and (start is None or end is None):
        raise ValueError("Provide both start and end.")
    start = _as_time(start, "start") if start is not None else None
    end = _as_time(end, "end") if end is not None else None
    if start is not None and end is not None and end <= start:
        raise ValueError("end must be after start.")
    return start, end


def _filename_bounds(path):
    match = _FILENAME_TIMES.match(Path(path).name)
    if match is None:
        raise WaveformError(
            f"Cannot infer the end time from {path}. "
            "Pass a DataFrame with local_path and time columns to specify each file's end time."
        )
    try:
        start, end = (pd.Timestamp(datetime.strptime(value, "%Y%m%d_%H%M%S")) for value in match.groups())
    except ValueError as exc:
        raise WaveformError(f"Invalid filename timestamps in {path}.") from exc
    if end < start:
        raise WaveformError(f"Filename end precedes start in {path}.")
    return start, end


def find_root_files(connection, *, device, channel, start, end):
    """Return all indexed files potentially overlapping [start, end).

    connection is an existing libs.sql_api.SQLConnection. device is the base
    name stored in root_files.device, e.g. 'caen8ch'; channel is 6 or '6'.
    Returned columns include the download identifiers, the database end-time
    anchor (time), and file_start/file_end parsed from filenames. No ROOT
    files are opened or downloaded. An empty search returns an empty table.
    """
    start, end = _window(start, end, required=True)
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty base device name, e.g. 'caen8ch'.")
    if isinstance(channel, bool) or not isinstance(channel, (str, numbers.Integral)):
        raise ValueError("channel must be a non-negative integer or a channel label.")
    channel = str(channel).strip().upper()
    if not re.fullmatch(r"(?:[0-9]+|[A-Z])", channel):
        raise ValueError("channel must be a non-negative integer or a channel label.")

    files = read_sql(_FIND_FILES_SQL, connection, params={
        "device": device.strip(), "file_pattern": f"%CH{channel}@%", "start": start, "end": end,
    })
    missing = set(_FILE_COLUMNS) - set(files.columns)
    if missing:
        raise WaveformError(f"File index result is missing columns: {', '.join(sorted(missing))}.")
    files = files[_FILE_COLUMNS].copy()
    for column in ("time", "file_start", "file_end"):
        files[column] = pd.to_datetime([_as_time(value, column) for value in files[column]])

    # Duplicate registrations must not double-count pulses. Conflicting end
    # anchors for the SAME file cannot be resolved safely, so report them.
    identity = ["computer", "dir", "file"]
    if (files.groupby(identity, dropna=False)["time"].nunique() > 1).any():
        raise WaveformError("The file index contains conflicting end times for the same file.")
    return files.drop_duplicates(identity).sort_values(
        ["file_start", "time", "computer", "dir", "file"], kind="stable"
    ).reset_index(drop=True)


def _local_records(files):
    if isinstance(files, pd.DataFrame):
        items = files.to_dict("records")
    elif isinstance(files, (str, os.PathLike, Mapping)):
        items = [files]
    else:
        items = list(files)

    records, seen = [], {}
    for item in items:
        record = dict(item) if isinstance(item, Mapping) else {"local_path": item}
        if "local_path" not in record:
            raise ValueError("Each file record must contain local_path.")
        path = Path(record["local_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ROOT file not found: {path}")
        if path.suffix.lower() not in {".root", ".root2"}:
            raise ValueError(f"Expected a .root or .root2 file: {path}")
        anchor = _as_time(record["time"], "file end time") if "time" in record else _filename_bounds(path)[1]
        key = path.resolve()
        if key in seen:
            if seen[key] != anchor:
                raise WaveformError(f"Conflicting end times supplied for {path}.")
            continue
        seen[key] = anchor
        records.append({**record, "local_path": str(path), "time": anchor})
    return records


def _timestamps(values, path):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
        raise WaveformError(f"Timestamp must contain non-negative integer picoseconds in {path}.")
    return values.astype(np.uint64, copy=False)


def _empty_pulses():
    return pd.DataFrame({
        "Timestamp": pd.Series(dtype="uint64"),
        "Energy": pd.Series(dtype="float64"),
        "Waveform": pd.Series(dtype="object"),
        "SourceFile": pd.Series(dtype="object"),
    }, index=pd.DatetimeIndex([], name="time"))


def read_waveforms(files, *, start=None, end=None, step_size="50 MB"):
    """Read local ROOT files into a pulse DataFrame without SQL or HTTP calls.

    files can be one path, an iterable of paths, or a DataFrame with local_path
    and time columns. time specifies each file's LAST-pulse calendar time.
    Bare paths use the end time encoded in their processed filename; explicit
    metadata also supports files without those filename prefixes.

    Optional start/end restrict the reconstructed times to [start, end).
    Omitting both reads all pulses. ROOT samples are read in chunks controlled
    by uproot's step_size, not loaded for every file at once.

    The result has a DatetimeIndex named time and Timestamp, Energy, Waveform,
    SourceFile columns. EnergyShort and PSP appear when any file has that
    branch; unavailable/undefined PSP values are NaN. Waveform cells hold
    NumPy arrays. Original Timestamp values are retained in picoseconds.
    SourceFile and df.attrs['root_files'] identify the local files and anchors
    so they can be inspected or passed back to this function for offline use.
    Repeated paths are read once, but pulses are NEVER deduplicated by time:
    different events can legitimately share the same pandas timestamp.
    """
    start, end = _window(start, end)
    records = _local_records(files)
    frames = []
    have_short = False
    for record in records:
        path, anchor = record["local_path"], record["time"]
        try:
            with uproot.open(path) as root_file:
                if "Data_R" not in root_file:
                    raise WaveformError(f"Data_R tree is missing in {path}.")
                tree = root_file["Data_R"]
                required = {"Timestamp", "Energy", "Samples"}
                missing = required - set(tree.keys())
                if missing:
                    raise WaveformError(f"Missing branches in {path}: {', '.join(sorted(missing))}.")
                has_short = "EnergyShort" in tree.keys()
                have_short |= has_short
                # Only the lightweight timestamp branch is scanned in full to
                # locate the latest pulse. Samples are subsequently chunked.
                timestamps = _timestamps(tree["Timestamp"].array(library="np"), path)
                if not len(timestamps):
                    continue
                latest = np.uint64(timestamps.max())
                branches = ["Timestamp", "Energy", "Samples"] + (["EnergyShort"] if has_short else [])
                for arrays in tree.iterate(branches, step_size=step_size, library="np"):
                    ts = _timestamps(arrays["Timestamp"], path)
                    # Subtract unsigned picoseconds in the safe direction,
                    # then ceil the lag to ns so absolute times round DOWN.
                    # No float seconds or signed-int64 epoch picoseconds are
                    # involved: both can lose precision or overflow.
                    lag_ps = latest - ts
                    lag_ns = (lag_ps // 1000 + (lag_ps % 1000 != 0)).astype(np.int64)
                    times = pd.DatetimeIndex(anchor - pd.to_timedelta(lag_ns, unit="ns"), name="time")
                    keep = np.ones(len(ts), dtype=bool)
                    if start is not None:
                        keep &= times >= start
                    if end is not None:
                        keep &= times < end
                    if not keep.any():
                        continue
                    # Copy only selected samples; retaining a view of a large
                    # chunk could otherwise keep unselected data in memory.
                    waveforms = [np.asarray(arrays["Samples"][i]).copy() for i in np.flatnonzero(keep)]
                    frame = pd.DataFrame({
                        "Timestamp": ts[keep], "Energy": arrays["Energy"][keep],
                        "SourceFile": path,
                    }, index=times[keep])
                    frame["Waveform"] = pd.Series(waveforms, index=frame.index, dtype=object)
                    if has_short:
                        frame["EnergyShort"] = arrays["EnergyShort"][keep]
                        energy = frame["Energy"].to_numpy(dtype=float)
                        short = frame["EnergyShort"].to_numpy(dtype=float)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            psp = 1.0 - short / energy
                        frame["PSP"] = np.where(np.isfinite(psp), psp, np.nan)
                    frames.append(frame)
        except WaveformError:
            raise
        except Exception as exc:
            raise WaveformError(f"Could not read waveforms from {path}: {exc}") from exc

    result = pd.concat(frames).sort_index(kind="stable") if frames else _empty_pulses()
    # Retain the optional schema even when the selected interval contains no
    # pulses from the file that provided EnergyShort.
    if have_short:
        for column in ("EnergyShort", "PSP"):
            if column not in result:
                result[column] = np.nan
    columns = ["Timestamp", "Energy", "Waveform"]
    columns += [column for column in ("EnergyShort", "PSP") if column in result]
    result = result[columns + ["SourceFile"]]
    result.attrs.update(root_files=records, start=start, end=end, timing_convention="per-file end-time anchor")
    return result


def _show_download_progress(received, total):
    megabytes = received / 1024**2
    message = f"Download {100 * received / total:.1f}% ({megabytes:.1f} MB)" if total else f"Downloaded {megabytes:.1f} MB"
    print(message, end="\r", flush=True)


def load_waveforms(
    connection,
    *,
    device,
    channel,
    start,
    end,
    download_dir="downloads/root_files",
    base_url=DEFAULT_BASE_URL,
    timeout=(10, 600),
    step_size="50 MB",
    show_progress=True,
):
    """Find files, download originals, and return all pulses in [start, end).

    Pass the SQLConnection already configured in the notebook; it handles
    database authentication and is left open for the caller to reuse/close.
    File transfers go through root_files_api.download_root_file. Each call
    downloads afresh; to reuse saved files, call read_waveforms instead.

    Files are saved in separate subdirectories derived from their complete
    index identity, preventing equal basenames on different computers/runs
    from overwriting one another. The filenames themselves are preserved.
    See read_waveforms for the output schema and this module's timing notes.
    No matching files raises FileNotFoundError; failures are never skipped.
    """
    start, end = _window(start, end, required=True)
    files = find_root_files(connection, device=device, channel=channel, start=start, end=end)
    if files.empty:
        raise FileNotFoundError(f"No indexed ROOT files overlap {start} to {end} for {device}, channel {channel}.")
    downloaded = []
    for number, row in enumerate(files.to_dict("records"), start=1):
        identity = "\0".join(row[key] for key in ("computer", "dir", "file"))
        subdirectory = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        if show_progress:
            print(f"Downloading {number}/{len(files)}: {row['file']}")
        path = download_root_file(
            row, download_dir=Path(download_dir) / subdirectory, base_url=base_url, timeout=timeout,
            progress=_show_download_progress if show_progress else None,
        )
        if show_progress:
            print()
        downloaded.append({**row, "local_path": str(path)})
    return read_waveforms(pd.DataFrame(downloaded), start=start, end=end, step_size=step_size)
