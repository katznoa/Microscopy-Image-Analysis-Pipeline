"""Quantify X-channel signal per cell: mean intensity + pixel histogram."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops

ARTIFACT_INTENSITY_PERCENTILE = 99.5  # threshold to find bright connected blobs --
# must stay a relative (per-image) percentile, NOT the pixel's exact max value: the
# saturated core of a real artifact blob is only a few px wide, while the visible
# halo sits mostly just below full saturation, so an exact-max threshold shatters
# each blob into dozens of disconnected 1-6px fragments and catches nothing
# (confirmed 2026-07-14 -- 0/4 known GFP artifacts detected that way).
ARTIFACT_DIAMETER_THRESHOLD = 35  # px -- real single-molecule spots/clusters in this
# dataset top out around ~24px equivalent diameter; large round saturated blobs
# (confirmed 41-63px, fully clipped at 65535) are non-biological debris/artifacts,
# not real signal. Gap between the two populations is clean (2026-07-14, on
# GFP ID 24 slide 4 slice 7 pic 1.czi and CRISPRa ID 10 slide 4 slice 1 pic 1.czi).
# No shape (circularity/solidity) check -- raw circularity on thresholded noisy
# pixel boundaries came out surprisingly low (0.21-0.36) even for confirmed round
# artifacts, so a naive shape filter would have wrongly let them back in.


def find_artifact_mask(x_channel, diameter_thresh=ARTIFACT_DIAMETER_THRESHOLD,
                        intensity_percentile=ARTIFACT_INTENSITY_PERCENTILE):
    """Boolean mask, True where a pixel belongs to an oversized bright blob
    (non-biological artifact) that should be excluded from quantification."""
    thresh = np.percentile(x_channel, intensity_percentile)
    bright = x_channel > thresh
    labeled, n = ndi.label(bright)
    mask = np.zeros(x_channel.shape, dtype=bool)
    for p in regionprops(labeled):
        if p.equivalent_diameter >= diameter_thresh:
            mask[labeled == p.label] = True
    return mask


def mask_out_artifacts(cell_masks, artifact_mask):
    """Return a copy of cell_masks with artifact pixels zeroed out (excluded from
    any downstream X-channel quantification), leaving GAPDH/dCas-based cell
    classification untouched since that uses the original cell_masks."""
    cleaned = cell_masks.copy()
    cleaned[artifact_mask] = 0
    return cleaned


def add_mean_x(df, cell_masks, x_channel):
    means = {}
    for lbl in df.label:
        px = x_channel[cell_masks == lbl]
        means[lbl] = px.mean() if px.size else 0.0
    df = df.copy()
    df["mean_X"] = df["label"].map(means)
    return df


def per_cell_histogram(cell_masks, x_channel, label, bins=50, value_range=None):
    px = x_channel[cell_masks == label]
    if value_range is None:
        value_range = (0, x_channel.max())
    counts, edges = np.histogram(px, bins=bins, range=value_range)
    return counts, edges


def pooled_histogram(cell_masks, x_channel, labels, bins=50, value_range=None):
    px = x_channel[np.isin(cell_masks, labels)]
    if value_range is None:
        value_range = (0, x_channel.max())
    counts, edges = np.histogram(px, bins=bins, range=value_range)
    return counts, edges
