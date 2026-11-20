# Libs

Reusable pieces of code that are used in several places.

`sql_api.py` runs SQL queries through the HTTPS SQL API.

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

Run the client tests from the repository root with:

```sh
python -m unittest discover -s tests -v
```
