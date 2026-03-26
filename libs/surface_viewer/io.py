from __future__ import annotations

from urllib.parse import parse_qs, quote, urljoin, urlparse

import pandas as pd
import requests
from tqdm.auto import tqdm


def get_roi_name_from_api_url(api_url: str) -> str:
    """Extract ROI name from the ROI API URL query parameter ``name``."""
    p = urlparse(api_url)
    qs = parse_qs(p.query)
    roi_name = (qs.get("name") or [None])[0]
    if not roi_name:
        raise ValueError("Could not find 'name' parameter in the ROI API URL.")
    return roi_name


def load_roi_api(api_url: str) -> pd.DataFrame:
    """Load ROI selections from the surface-viewer ROI API into a DataFrame."""
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    data = r.json()
    selections = data.get("selections", [])
    df = pd.DataFrame(selections)

    if df.empty:
        return df

    for c in ["row", "col"]:
        if c in df.columns:
            df[c] = df[c].astype("int64")

    for c in ["srcJson", "basename", "foldername"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


def infer_dataset_base_from_api(api_url: str) -> str:
    """Infer the dataset base URL ending in ``/`` from an ROI API URL."""
    p = urlparse(api_url)
    qs = parse_qs(p.query)
    dataset = (qs.get("dataset") or [None])[0]
    if not dataset:
        raise ValueError("Could not find 'dataset' parameter in the API URL.")

    root = f"{p.scheme}://{p.netloc}/surface-viewer/data/"
    dataset_encoded = quote(dataset, safe="")
    return urljoin(root, dataset_encoded + "/")


def add_json_urls(df: pd.DataFrame, api_url: str) -> pd.DataFrame:
    """Add a ``json_url`` column by resolving ``srcJson`` against the dataset base."""
    if df.empty:
        df = df.copy()
        df["json_url"] = pd.Series(dtype="string")
        return df

    dataset_base = infer_dataset_base_from_api(api_url)
    df = df.copy()
    df["json_url"] = df["srcJson"].apply(lambda p: urljoin(dataset_base, str(p)))
    return df


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "eds-demo-notebook/0.1"})
    return s


def fetch_json_items(url: str, session: requests.Session | None = None) -> list:
    """GET a JSON file and return a list of records from either a list or ``{'items': ...}``."""
    session = session or _new_session()
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def build_spectrum_index(
    urls: list[str],
    *,
    progress: bool = True,
    session: requests.Session | None = None,
) -> dict[tuple[str, int, int], list[int]]:
    """Build a lookup ``(json_url, row, col) -> spectrum`` by reading each JSON file once."""
    session = session or _new_session()
    index: dict[tuple[str, int, int], list[int]] = {}

    for url in tqdm(urls, desc="Downloading JSON files", disable=not progress):
        items = fetch_json_items(url, session)
        for rec in items:
            r = rec.get("rownum", rec.get("row"))
            c = rec.get("colnum", rec.get("col"))
            spec = rec.get("aggregatedspectrum") or rec.get("aggregatedSpectrum") or rec.get("spectrum")
            if r is None or c is None or spec is None:
                continue
            try:
                key = (url, int(r), int(c))
                index[key] = [int(x) for x in spec]
            except Exception:
                pass

    return index


def attach_spectra(
    df: pd.DataFrame,
    index: dict[tuple[str, int, int], list[int]],
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """Add a ``spectrum`` column to a DataFrame using the pre-built spectrum index."""

    def pick(row):
        return index.get((row["json_url"], int(row["row"]), int(row["col"])), None)

    df = df.copy()
    if progress:
        tqdm.pandas(desc="Indexing spectra")
        df["spectrum"] = df.progress_apply(pick, axis=1)
    else:
        df["spectrum"] = df.apply(pick, axis=1)
    return df


def get_selection_grid_url(api_url: str) -> str:
    """Return the first matching selection-grid JSON URL from the dataset overlays folder."""
    dataset_base = infer_dataset_base_from_api(api_url)
    candidates = [
        "overlays/selection-grid.json",
        "overlays/selection_grid.json",
        "selection-grid.json",
        "selection_grid.json",
    ]
    session = _new_session()
    for rel in candidates:
        url = urljoin(dataset_base, rel)
        try:
            r = session.get(url, timeout=30)
            if r.ok:
                return url
        except Exception:
            pass
    raise FileNotFoundError("Could not find selection-grid.json in the dataset overlays folder.")


def load_all_cells_from_selection_grid(api_url: str) -> pd.DataFrame:
    """Load the full cell grid from ``overlays/selection-grid.json`` into a DataFrame."""
    grid_url = get_selection_grid_url(api_url)
    session = _new_session()
    items = fetch_json_items(grid_url, session)

    rows = []
    for rec in items:
        if rec.get("type") != "rect":
            continue
        r = rec.get("rownum", rec.get("row"))
        c = rec.get("colnum", rec.get("col"))
        src = rec.get("srcJson")
        if r is None or c is None or src is None:
            continue
        rows.append({
            "row": int(r),
            "col": int(c),
            "srcJson": str(src),
            "basename": rec.get("basename"),
            "label": rec.get("label"),
            "x": rec.get("x"),
            "y": rec.get("y"),
            "width": rec.get("width"),
            "height": rec.get("height"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["srcJson", "basename", "label"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    df = add_json_urls(df, api_url)
    df = df.drop_duplicates(subset=["json_url", "row", "col"]).reset_index(drop=True)
    return df
