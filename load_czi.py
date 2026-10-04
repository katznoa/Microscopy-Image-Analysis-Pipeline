"""Load a Zeiss CZI file and pull out the 4 Airyscan-processed channels we care about."""
from pathlib import Path

import numpy as np
from aicspylibczi import CziFile

# CZI channel name -> biological target (confirmed with user 2026-07-13)
CHANNEL_MAP = {
    "AF647-T1": "X",
    "AF546-T2": "dCas",
    "AF488-T3": "GAPDH",
    "DAPI-T4": "DAPI",
}

# Confirmed with user 2026-07-13 against their Fiji workflow: keep channel names
# WITHOUT '#' (Gray16, Detector: Airyscan) — Fiji shows these as C=1,3,5,7 (1-indexed).
USE_HASH_CHANNELS = False


def channel_name(base_name):
    return f"{base_name}#" if USE_HASH_CHANNELS else base_name


def load_airyscan_channels(czi_path, h_index=0):
    """Return dict: target name -> 2D array (Z-max-projected), using H=h_index sub-frame."""
    czi = CziFile(str(czi_path))

    # map channel *name* -> C index, from the czi's own channel metadata
    name_to_cindex = {}
    for el in czi.meta.iter("Channel"):
        name = el.attrib.get("Name")
        if name is not None and "Id" in el.attrib:
            cidx = int(el.attrib["Id"].split(":")[1])
            name_to_cindex[name] = cidx

    dims_shape = czi.get_dims_shape()[0]
    n_z = dims_shape["Z"][1] - dims_shape["Z"][0]

    result = {}
    for base_name, target in CHANNEL_MAP.items():
        cname = channel_name(base_name)
        cidx = name_to_cindex[cname]
        zplanes = []
        for z in range(n_z):
            arr, _ = czi.read_image(C=cidx, Z=z, H=h_index, T=0)
            zplanes.append(arr.squeeze())
        stack = np.stack(zplanes, axis=0)
        result[target] = stack.max(axis=0)  # max-intensity project

    return result


def read_all_h_frames(czi_path, base_name="AF647-T1", z=1):
    """Debug helper: return all 4 H sub-frames for one channel/Z, for visual inspection."""
    czi = CziFile(str(czi_path))
    name_to_cindex = {}
    for el in czi.meta.iter("Channel"):
        name = el.attrib.get("Name")
        if name is not None and "Id" in el.attrib:
            cidx = int(el.attrib["Id"].split(":")[1])
            name_to_cindex[name] = cidx
    cidx = name_to_cindex[base_name]
    frames = {}
    for h in range(4):
        arr, _ = czi.read_image(C=cidx, Z=z, H=h, T=0)
        frames[h] = arr.squeeze()
    return frames
