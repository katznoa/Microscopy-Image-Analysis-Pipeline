"""Nuclei-seeded whole-cell segmentation from DAPI + GAPDH.

Validated approach (2026-07-13/14, on CRISPRa ID 10 slide 4 slice 1 pic 1.czi):
1. Segment nuclei from DAPI with Cellpose, diameter ~60px at this pixel size.
   IMPORTANT: run per-tile (~800x800), not on the full image at once. Cellpose's
   global percentile normalization on the full frame washes out local contrast and
   under-segments touching nuclei that split cleanly on a small, well-normalized
   crop -- confirmed by comparing an isolated 800x800 crop (correctly split touching
   nuclei) against the same region run as part of the full 3233x3233 image (merged
   into one blob). `normalize={"tile_norm_blocksize": ...}` was tried and did not
   fix this. Tiling manually and stitching by keeping only detections centered in
   each tile's non-overlap "core" region reproduces the validated crop-level result.
2. Grow each nucleus into a whole-cell region via marker-controlled watershed over
   blurred GAPDH signal (punctate FISH-style spots, not a smooth cytoplasm stain --
   raw GAPDH can't be thresholded directly into a cell mask).
3. Tight-ish GAPDH threshold + small morphological closing to match manual
   annotation style (boundary hugs the nucleus + immediate dense signal, doesn't
   reach far for sparse distant puncta), then fill interior holes per cell.

A single joint Cellpose model on DAPI+GAPDH together was tried first and rejected --
it under-segments touching cells (merges neighboring nuclei) because it isn't
anchored to the actual nucleus count.
"""
from pathlib import Path

import numpy as np
from cellpose import models
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import closing, dilation, disk
from skimage.segmentation import watershed

NUCLEI_DIAMETER = 60  # px, at ~0.038 um/px Airyscan pixel size
GAPDH_BLUR_SIGMA = 8  # px, smooths puncta into a density map
GAPDH_THRESHOLD_FRACTION = 1.0  # fraction of Otsu threshold on blurred GAPDH
CLOSING_RADIUS = 10  # px, closes gaps from dim/sparse puncta
MAX_CELL_RADIUS = 120  # px, cap on how far a cell can grow from its nucleus
GRAYSCALE_CLOSE_RADIUS = 20  # px, fills local dark dips in GAPDH signal (e.g. a
# vacuole-like gap in the middle of a cell) that would otherwise act as a false
# watershed "ridge" and route a cell-cell boundary straight through real cytoplasm

TILE_SIZE = 800  # px, matches the validated standalone-crop test
TILE_OVERLAP = 200  # px, margin used to avoid double-counting nuclei at tile edges
MIN_NUCLEUS_AREA = 3000  # px, rejects tiny false-positive fragments -- flow_threshold=0.6
# (needed to catch real dim nuclei, see module docstring) can over-fragment in some
# tissue/images, producing small sub-region "nuclei" sitting inside a real, larger
# nucleus. Confirmed visually on GFP ID 24 slide 4 slice 7 pic 1 (2026-07-14).

_model = None


def get_model():
    global _model
    if _model is None:
        _model = models.CellposeModel(gpu=True)
    return _model


def _tile_bounds(size, tile_size, overlap):
    step = tile_size - overlap
    starts = list(range(0, max(size - tile_size, 0) + 1, step))
    if not starts or starts[-1] + tile_size < size:
        starts.append(max(size - tile_size, 0))
    return starts


def segment_nuclei_tiled(dapi, diameter=NUCLEI_DIAMETER, tile_size=TILE_SIZE,
                          overlap=TILE_OVERLAP, verbose=False, overlap_thresh=0.3,
                          min_area=MIN_NUCLEUS_AREA):
    """Segment nuclei tile-by-tile (each tile gets its own local normalization,
    matching the validated small-crop behavior) and stitch into one label image.

    Dedup strategy: collect every candidate nucleus from every tile (a nucleus near
    a boundary is typically seen by 2-4 overlapping tiles), rank by distance from
    its own tile's center (central detections are the most reliable -- Cellpose
    degrades near tile edges), then greedily accept candidates in that order,
    skipping any that substantially overlap an already-accepted nucleus. This is
    more robust than requiring one specific "owning" tile to get the detection
    right -- if that tile misses/merges it, a neighboring tile's clean detection
    still wins instead of the nucleus being lost entirely.
    """
    model = get_model()
    H, W = dapi.shape
    y_starts = _tile_bounds(H, tile_size, overlap)
    x_starts = _tile_bounds(W, tile_size, overlap)

    candidates = []  # (confidence, global_ys, global_xs)
    for yi, y0 in enumerate(y_starts):
        for xi, x0 in enumerate(x_starts):
            y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
            tile = dapi[y0:y1, x0:x1].astype(float)
            masks, _, _ = model.eval(
                tile, channel_axis=None, diameter=diameter,
                flow_threshold=0.6, cellprob_threshold=0.0,
            )
            if verbose:
                print(f"tile ({yi},{xi}) [{y0}:{y1},{x0}:{x1}]: {masks.max()} nuclei")

            th, tw = y1 - y0, x1 - x0
            tile_cy, tile_cx = th / 2, tw / 2
            for lbl in range(1, masks.max() + 1):
                ys, xs = np.where(masks == lbl)
                if len(ys) < min_area:
                    continue
                dist = ((ys.mean() - tile_cy) ** 2 + (xs.mean() - tile_cx) ** 2) ** 0.5
                candidates.append((dist, ys + y0, xs + x0))

    candidates.sort(key=lambda c: c[0])

    result = np.zeros((H, W), dtype=np.int32)
    next_label = 1
    for dist, ys, xs in candidates:
        existing = result[ys, xs]
        overlap_frac = (existing != 0).mean()
        if overlap_frac > overlap_thresh:
            continue
        keep = existing == 0
        result[ys[keep], xs[keep]] = next_label
        next_label += 1

    if verbose:
        print(f"Deduped {len(candidates)} candidates -> {next_label - 1} nuclei")

    return result


def segment_nuclei(dapi, diameter=NUCLEI_DIAMETER):
    model = get_model()
    masks, _, _ = model.eval(
        dapi.astype(float), channel_axis=None, diameter=diameter,
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks


def segment_cells(dapi, gapdh, nuclei_masks=None):
    """Return (nuclei_masks, cell_masks) -- both label images, same labeling."""
    if nuclei_masks is None:
        nuclei_masks = segment_nuclei_tiled(dapi)

    gapdh_blur = gaussian(gapdh.astype(float), sigma=GAPDH_BLUR_SIGMA, preserve_range=True)

    # grayscale closing (not binary) fills local dark dips in the intensity itself,
    # so a small gap surrounded by real signal doesn't create a false low-elevation
    # "ridge" for watershed to route a cell-cell boundary through -- fixes cells
    # getting incorrectly split down the middle by a dim spot in their own cytoplasm
    elevation_source = closing(gapdh_blur, disk(GRAYSCALE_CLOSE_RADIUS))

    thresh = threshold_otsu(elevation_source) * GAPDH_THRESHOLD_FRACTION
    signal_mask = elevation_source > thresh
    signal_mask = closing(signal_mask, disk(CLOSING_RADIUS))

    reach_mask = dilation(nuclei_masks > 0, disk(MAX_CELL_RADIUS))
    allowed = (signal_mask | (nuclei_masks > 0)) & reach_mask

    elevation = -elevation_source
    cell_masks = watershed(elevation, markers=nuclei_masks, mask=allowed)

    # fill interior holes, but only into true background (0) -- never overwrite a
    # neighboring label's already-assigned pixels (that previously let a larger
    # label's hole-fill silently swallow a smaller adjacent label's entire region)
    filled = cell_masks.copy()
    background = cell_masks == 0
    for lbl in range(1, nuclei_masks.max() + 1):
        hole = ndi.binary_fill_holes(cell_masks == lbl) & background
        filled[hole] = lbl
    cell_masks = filled

    return nuclei_masks, cell_masks
