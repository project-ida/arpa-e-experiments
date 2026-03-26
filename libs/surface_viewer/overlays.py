from __future__ import annotations

from getpass import getpass

import requests
from requests.auth import HTTPBasicAuth

API_URL = "https://nucleonics.mit.edu/surface-viewer/api/protected/create_heatmap_overlay.php"


def get_api_auth(username=None, password=None):
    """Prompt interactively if username/password are not provided."""
    if username is None:
        username = getpass("API username: ")
    if password is None:
        password = getpass("API password: ")
    return HTTPBasicAuth(username, password)


def create_overlay(
    auth,
    dataset,
    input_folder,
    band_start,
    band_end,
    vmin,
    vmax,
    threshold=None,
    palette=None,
    fill=None,
    dry_run=False,
    timeout=300,
    api_url=API_URL,
    verbose=True,
):
    """Create a heatmap overlay on the server."""
    payload = {
        "action": "create",
        "dataset": dataset,
        "input_folder": input_folder,
        "band_start": int(band_start),
        "band_end": int(band_end),
        "vmin": int(vmin),
        "vmax": int(vmax),
        "dry_run": bool(dry_run),
    }

    if threshold is not None:
        payload["threshold"] = int(threshold)
    if palette is not None:
        payload["palette"] = str(palette)
    if fill is not None:
        payload["fill"] = str(fill)

    r = requests.post(api_url, json=payload, auth=auth, timeout=timeout)

    if verbose:
        print(f"Status: {r.status_code}")

    try:
        data = r.json()
    except Exception:
        if verbose:
            print("Non-JSON response:")
            print(r.text[:4000])
        r.raise_for_status()
        raise RuntimeError("Server returned a non-JSON response.")

    if verbose:
        print(data)

    if not r.ok or not data.get("ok", False):
        raise RuntimeError(f"Overlay creation failed: {data}")

    return data


def delete_overlay(
    auth,
    dataset,
    overlay_file,
    timeout=300,
    api_url=API_URL,
    verbose=True,
):
    """Delete an existing heatmap overlay on the server."""
    payload = {
        "action": "delete",
        "dataset": dataset,
        "overlay_file": overlay_file,
    }

    r = requests.post(api_url, json=payload, auth=auth, timeout=timeout)

    if verbose:
        print(f"Status: {r.status_code}")

    try:
        data = r.json()
    except Exception:
        if verbose:
            print("Non-JSON response:")
            print(r.text[:4000])
        r.raise_for_status()
        raise RuntimeError("Server returned a non-JSON response.")

    if verbose:
        print(data)

    if not r.ok or not data.get("ok", False):
        raise RuntimeError(f"Overlay deletion failed: {data}")

    return data
