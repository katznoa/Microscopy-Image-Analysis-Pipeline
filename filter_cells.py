"""Filter segmented cells: drop GAPDH-inactive cells, fused/overlapping nuclei,
and cells whose whole-cell mask never grew past the nucleus.

Per user's manual annotation reference (2026-07-14): cells with DAPI but no real
GAPDH signal ("inactive") are excluded entirely, as are nuclei that look like two
overlapping cells fused into one blob.

GAPDH-active and dCas-positive thresholds use FIXED absolute values (2026-07-14),
not per-image Otsu -- Otsu splits each image's own distribution regardless of
where real biological background actually sits, giving a different, non-comparable
cutoff per image. Fixed values keep classification consistent across all 24 files.

MIN_CELL_AREA (2026-07-17): the whole-cell watershed mask sometimes never grows
beyond the nucleus itself (little/no surrounding GAPDH signal for the watershed
to expand into), producing a "cell" that's really just a bare nucleus. Visual QC
confirmed this at 5 of 6 sampled masks <=4200px, and it persists in a shrinking
but nonzero fraction up through ~18000px -- there's no size at which it fully
disappears, since it depends on local GAPDH availability, not cell size. 8000px
was chosen as a conservative cutoff: it removes the clearest bare-nucleus cases
(~5% of previously-kept cells) without the much larger data loss (~15%) a higher
cutoff like 18000-20000px would cost. Note the cutoff isn't neutral across
conditions -- GFP (no dCas filter, so more small masks survive) sees a bigger
share of its cells trimmed, and its mean_X shifts upward as a result (removed
cells were its dimmest). (Briefly tried 4000/4500 on 2026-07-18; reverted to
8000 -- the original, visually-validated cutoff.)
"""
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table

GAPDH_ACTIVE_THRESHOLD = 1000
DCAS_POSITIVE_THRESHOLD = 6000  # updated 2026-09-21 (was 8000)
MIN_CELL_AREA = 8000  # px^2, whole-cell (watershed) mask minimum -- see module docstring


def _mean_intensity_per_cell(cell_masks, labels, channel):
    means = {}
    for lbl in labels:
        px = channel[cell_masks == lbl]
        means[lbl] = px.mean() if px.size > 0 else 0.0
    return pd.Series(means)


def compute_cell_features(nuclei_masks, cell_masks, gapdh, dcas=None):
    """One row per nucleus label: nucleus_area, solidity, eccentricity, mean_gapdh,
    and mean_dcas (if dcas channel provided), all measured within the cell mask.
    Also includes cell_area -- the whole-cell (watershed) mask size, which can
    differ substantially from nucleus area when watershed growth stalls near a
    nucleus with little surrounding GAPDH signal."""
    props = regionprops_table(
        nuclei_masks, properties=["label", "area", "solidity", "eccentricity"]
    )
    df = pd.DataFrame(props).set_index("label")

    cell_labels, cell_counts = np.unique(cell_masks, return_counts=True)
    cell_area = pd.Series(cell_counts, index=cell_labels).drop(0, errors="ignore")
    df["cell_area"] = cell_area.reindex(df.index).fillna(0).astype(int)

    df["mean_gapdh"] = _mean_intensity_per_cell(cell_masks, df.index, gapdh)
    if dcas is not None:
        df["mean_dcas"] = _mean_intensity_per_cell(cell_masks, df.index, dcas)

    return df.reset_index()


def classify_by_otsu(df, col):
    """Otsu threshold on a per-cell mean-intensity column -> boolean 'positive' column."""
    values = df[col].values
    thresh = threshold_otsu(values)
    return df[col] >= thresh, thresh


def classify_by_fixed(df, col, threshold):
    """Fixed absolute threshold on a per-cell mean-intensity column."""
    return df[col] >= threshold, threshold


def classify_overlap(df, area_col="area", solidity_col="solidity",
                      area_z_thresh=2.0, solidity_thresh=0.90):
    """Flag nuclei that look like two fused/overlapping cells: unusually large area
    (z-score vs population median/MAD) or low solidity (concave/irregular outline)."""
    median_area = df[area_col].median()
    mad_area = (df[area_col] - median_area).abs().median() * 1.4826  # MAD -> std estimate
    if mad_area == 0:
        area_flag = pd.Series(False, index=df.index)
    else:
        area_z = (df[area_col] - median_area) / mad_area
        area_flag = area_z > area_z_thresh
    solidity_flag = df[solidity_col] < solidity_thresh
    is_overlap = area_flag | solidity_flag
    return ~is_overlap, {"median_area": median_area, "mad_area": mad_area}


def filter_cells(nuclei_masks, cell_masks, gapdh, dcas=None,
                  gapdh_thresh=GAPDH_ACTIVE_THRESHOLD, dcas_thresh=DCAS_POSITIVE_THRESHOLD,
                  min_cell_area=MIN_CELL_AREA):
    """Full filter chain: GAPDH-active (real cell) AND not a fused/overlapping
    nucleus AND whole-cell mask big enough to be real cytoplasm (not just a bare
    nucleus where watershed growth stalled) AND (if dcas provided) dCas-positive
    (successful delivery).

    Both thresholds are fixed absolute values by default (see module docstring).
    Pass gapdh_thresh/dcas_thresh=None to fall back to per-image Otsu instead.
    """
    df = compute_cell_features(nuclei_masks, cell_masks, gapdh, dcas=dcas)

    if gapdh_thresh is None:
        df["active"], gapdh_thresh = classify_by_otsu(df, "mean_gapdh")
    else:
        df["active"], gapdh_thresh = classify_by_fixed(df, "mean_gapdh", gapdh_thresh)

    df["not_overlap"], overlap_stats = classify_overlap(df)
    df["cell_size_ok"] = df["cell_area"] >= min_cell_area
    df["real_cell"] = df["active"] & df["not_overlap"] & df["cell_size_ok"]

    info = {"gapdh_thresh": gapdh_thresh, "min_cell_area": min_cell_area, **overlap_stats}

    if dcas is not None:
        if dcas_thresh is None:
            real = df[df["real_cell"]]
            _, dcas_thresh = classify_by_otsu(real, "mean_dcas")
        df["dcas_positive"], dcas_thresh = classify_by_fixed(df, "mean_dcas", dcas_thresh)
        df.loc[~df["real_cell"], "dcas_positive"] = False
        df["keep"] = df["real_cell"] & df["dcas_positive"]
        info["dcas_thresh"] = dcas_thresh
    else:
        df["keep"] = df["real_cell"]

    return df, info
