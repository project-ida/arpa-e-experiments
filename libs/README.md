# Libs

Reusable pieces of code that are used in several places.

`sql_api.py` runs SQL queries through the HTTPS SQL API.

## Waveform DataFrames

`waveforms.py` finds every ROOT file overlapping an interval, downloads the
originals through `root_files_api.py`, and returns the pulses in one pandas
DataFrame. It accepts the existing SQL connection; credentials stay in the
notebook's connection setup.

```python
from libs.waveforms import load_waveforms

df_pulses = load_waveforms(
    connection,
    device="caen8ch",
    channel=6,
    start="2025-09-03 11:55:00",
    end="2025-09-03 12:05:00",
)
```

The interval includes `start` and excludes `end`. All pulses within that
interval are returned, ready for energy/PSP filtering and plotting. The index
is reconstructed absolute time; columns include the original picosecond
`Timestamp`, `Energy`, sample arrays in `Waveform`, and `SourceFile`. Optional
`EnergyShort` and calculated `PSP` columns contain NaN where unavailable or
undefined. Failed downloads or reads raise an error rather than omit a file.

Dependencies are `pandas`, `numpy`, `uproot`, and `requests`. Downloads go into
`downloads/root_files` with separate subdirectories for each indexed file,
preserving its filename. Set `show_progress=False` to suppress progress output.
Each `load_waveforms` call downloads afresh.

For files already downloaded, use `read_waveforms`. Paths with the processed
`YYYYMMDD_HHMMSS-YYYYMMDD_HHMMSS_` prefix supply their own end-time metadata:

```python
from libs.waveforms import read_waveforms

df_pulses = read_waveforms(
    ["path/to/20250903_115839-20250903_120201_DataR_CH6@test.root2"],
    start="2025-09-03 12:00:00",
    end="2025-09-03 12:01:00",
)
```

You can also save the file list and anchors returned by `load_waveforms` and
reuse them in a later session:

```python
import pandas as pd

pd.DataFrame(df_pulses.attrs["root_files"]).to_csv("waveform-files.csv", index=False)

# Later, using the saved local files:
saved_files = pd.read_csv("waveform-files.csv")
df_pulses = read_waveforms(saved_files, start=start, end=end)
```

Explicit file tables have `local_path` and `time` columns, where `time` is each
file's last-pulse time. Omitting `start` and `end` reads the entire supplied
files. `find_root_files(connection, device=..., channel=..., start=..., end=...)`
is available to inspect the matching index rows without downloading anything.

Timing conventions and their precision limits are documented at the top of
[`waveforms.py`](waveforms.py). The reader preserves the starter's per-file
end-time anchor. Filename/index anchors omit fractional seconds, so the
reconstructed index does not reproduce the original pulse-table timestamps
at subsecond precision. The original picosecond `Timestamp` column is retained.
Calendar inputs use the same naive local clock as the file index.

## Individual ROOT downloads

`root_files_api.py` downloads original ROOT files through the waveform viewer
on Nucleonics, which already has access to the Synology mount:

```python
from libs.root_files_api import download_root_file

# closest_root_file is a row from the root_files SQL table, converted to a dict.
filename = download_root_file(closest_root_file)
```

The only client dependency is `requests`. The default server is
`http://nucleonics.mit.edu:8080`; pass `base_url` to use another installation.
The server must have the `/root-files/download` route from `ida-scripts`.
Files go into `downloads/root_files` by default, and the returned path can be
opened with `uproot`. Each call downloads afresh and replaces an existing file
only after the transfer succeeds. An optional `progress(received_bytes, total_bytes)`
callback reports progress; the total can be `None`.

Run the download and waveform tests from the repository root with:

```sh
python -m unittest discover -s tests -v
```
