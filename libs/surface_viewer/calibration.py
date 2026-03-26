from __future__ import annotations

import re
from urllib.parse import urljoin

import numpy as np
import requests

from .io import infer_dataset_base_from_api


def load_config_txt(dataset_base: str) -> dict:
    """Parse ``config.txt`` from a dataset folder in the same way as the viewer."""
    url = urljoin(dataset_base, "config.txt")
    r = requests.get(url, timeout=30)
    if not r.ok:
        return {}

    cfg = {}
    for line in r.text.splitlines():
        s = re.sub(r"#.*$", "", line).strip()
        if not s or "=" not in s:
            continue
        k, v = s.split("=", 1)
        cfg[k.strip().lower()] = v.strip()
    return cfg


def get_energy_cal_from_dataset(
    api_url: str,
    default_eV_per_ch=20.000347,
    default_start_eV=-192.768,
):
    dataset_base = infer_dataset_base_from_api(api_url)
    cfg = load_config_txt(dataset_base)

    eV_per_ch = float(cfg.get("eds_ev_per_ch", default_eV_per_ch))
    start_eV = float(cfg.get("eds_start_ev", default_start_eV))

    n_channels = cfg.get("eds_n_channels", None)
    n_channels = int(n_channels) if n_channels and str(n_channels).isdigit() else None

    return {
        "dataset_base": dataset_base,
        "eV_per_ch": eV_per_ch,
        "start_eV": start_eV,
        "n_channels": n_channels,
        "raw_cfg": cfg,
    }


def make_energy_axis(cum, cal: dict):
    n = len(cum) if cal.get("n_channels") is None else min(len(cum), cal["n_channels"])
    return (cal["start_eV"] + np.arange(n) * cal["eV_per_ch"]) / 1000.0


def make_energy_axis_from_length(n, cal: dict):
    return (cal["start_eV"] + np.arange(n) * cal["eV_per_ch"]) / 1000.0


def channel_to_keV(ch, cal: dict):
    return (cal["start_eV"] + ch * cal["eV_per_ch"]) / 1000.0


def maybe_get_calibration(roi_api_urls, need_calibration=False, allow_defaults=True):
    if not need_calibration:
        return None
    first_url = roi_api_urls[0]
    cal = get_energy_cal_from_dataset(first_url)
    raw_cfg = cal.get("raw_cfg", {}) or {}
    cal["from_config"] = ("eds_ev_per_ch" in raw_cfg and "eds_start_ev" in raw_cfg)
    if not cal["from_config"] and not allow_defaults:
        return None
    return cal


def keV_to_channel(keV, cal: dict):
    return int(round((float(keV) * 1000.0 - cal["start_eV"]) / cal["eV_per_ch"]))
