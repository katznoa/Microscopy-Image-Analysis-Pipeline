"""Full pipeline for one CZI file: segment -> filter -> quantify -> export.
Usage: python process_image.py "<filename.czi>" --data-dir "<path>" --out-dir "<path>"
--data-dir/--out-dir are optional; default to the original 26_07_13 experiment paths.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.measure import find_contours

from load_czi import load_airyscan_channels
from segment import segment_nuclei_tiled, segment_cells
from filter_cells import filter_cells, GAPDH_ACTIVE_THRESHOLD, DCAS_POSITIVE_THRESHOLD, MIN_CELL_AREA
from quantify import add_mean_x, pooled_histogram, find_artifact_mask, mask_out_artifacts

DEFAULT_DATA_DIR = Path(
    "/Users/noakatz/Documents/microscopy_invivo_sections/26_07_13_all_probes_repeat"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "26_07_13_all_probes_repeat"

N_BINS = 100

# Images excluded from the main analysis (2026-07-17): whole-image DAPI/GAPDH/dCas/X
# in "IFFL ID 4 slide 3 slice 7" (pic 1-3) run ~40-70% brighter across every channel
# than "slide 3 slice 6" of the same construct+ID, uniformly across all cells in the
# image, not just a bright subpopulation -- confirmed via whole-image channel means
# and per-cell mean_gapdh both roughly doubling. Per user: pic 3 has unusually high
# GAPDH signal, skipped in their own manual (FIJI-based) annotation -- pic 3 remains
# excluded. Pic 1 has gone back and forth (2026-07-18): brought in, excluded again
# (mean_X ran noticeably higher than pic 2/rest of dataset), and is being brought
# back in once more on 2026-07-18 for another look. Check with user for current
# intent if this matters -- it's been unstable.
EXCLUDED_IMAGES = {
    "IFFL ID 4 slide 3 slice 7 pic 3",
    # Quantitative whole-image channel-mean outliers, flagged 2026-07-18: MAD
    # z-score (per-condition, robust) on GAPDH, dCas, AND X simultaneously
    # exceeds 1.9 for all three channels -- see scratch/out/whole_image_channel_means.xlsx
    # ("methods" sheet) for the full threshold derivation and justification.
    "IFFL ID 5 slide 3 slice 1 pic 3",
    "IFFL ID 5 slide 3 slice 2 pic 2",
    # Whole-session batch effect, confirmed 2026-07-22: GFP ID24 in the 26_07_13
    # ("July") session reads ~1.4x higher mean_X than the same construct+ID in
    # the 26_06_08 ("June") session (per-cell mean=1425 vs 1032, Mann-Whitney
    # p=4.8e-16), with GAPDH and X both elevated while dCas stays flat (~65-70,
    # near background either way, as expected for GFP) -- consistent with a
    # session-level acquisition/hybridization problem specific to this GFP
    # capture, not real biology. All 3 pics of this construct/session excluded.
    "GFP ID 24 slide 4 slice 7 pic 1",
    "GFP ID 24 slide 4 slice 7 pic 2",
    "GFP ID 24 slide 4 slice 7 pic 3",
}


def to_display(img, p_lo=1, p_hi=99.5):
    lo, hi = np.percentile(img, (p_lo, p_hi))
    return rescale_intensity(img.astype(float), in_range=(lo, hi), out_range=(0, 1))


def group_sheet_name_from_filename(path):
    """Sheet name grouping all pictures of the same physical slide/slice together
    (e.g. 'CRISPRa_ID9_sl5_sc1' covers pic 1, 2, and 3) -- per user, 2026-07-14."""
    stem = Path(path).stem
    stem = re.sub(r"\s*pic\s*\d+\s*$", "", stem)  # drop trailing "pic N"
    stem = (
        stem.replace("ID ", "ID")
        .replace("slide ", "sl")
        .replace("slice ", "sc")
        .replace(" ", "_")
    )
    return stem[:31]


# Filenames with no "pic N" suffix default to pic 1, which collides with an
# existing "pic 1" file in the same slide/slice group and silently overwrites it
# in the merged sheet. Override the resolved pic number for specific known cases
# here instead of renaming source files. Per user, 2026-07-18: "GFP ID 23 slide 2
# slice 6.czi" (26_05_28 dataset) is a real 4th picture of that slide/slice.
PIC_NUMBER_OVERRIDES = {
    "GFP ID 23 slide 2 slice 6": "4",
}


def pic_number_from_filename(path):
    stem = Path(path).stem
    if stem in PIC_NUMBER_OVERRIDES:
        return PIC_NUMBER_OVERRIDES[stem]
    m = re.search(r"pic\s*(\d+)\s*$", stem)
    return m.group(1) if m else "1"


def main(filename, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR,
         gapdh_thresh=GAPDH_ACTIVE_THRESHOLD, dcas_thresh=DCAS_POSITIVE_THRESHOLD,
         min_cell_area=MIN_CELL_AREA):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    czi_path = data_dir / filename
    stem = czi_path.stem
    if stem in EXCLUDED_IMAGES:
        print(f"Skipping {filename}: in EXCLUDED_IMAGES (see comment above that constant)")
        return
    group_sheet = group_sheet_name_from_filename(czi_path)
    pic_num = pic_number_from_filename(czi_path)

    # GFP-construct images have no dCas present -- skip the dCas-positive screen
    # and use all viable (GAPDH-active, non-overlapping) cells instead (per user,
    # 2026-07-14). Construct is the first word of the filename.
    construct = filename.split()[0].upper()
    use_dcas_filter = construct != "GFP"
    print(f"Processing {filename} (group sheet: {group_sheet}, pic={pic_num}, "
          f"construct={construct}, dcas_filter={'on' if use_dcas_filter else 'off'})")

    print("Loading channels...")
    channels = load_airyscan_channels(czi_path, h_index=0)
    dapi, gapdh, dcas, x = channels["DAPI"], channels["GAPDH"], channels["dCas"], channels["X"]

    nuclei_path = out_dir / f"{stem}_nuclei_masks.npy"
    cell_path = out_dir / f"{stem}_cell_masks.npy"
    if nuclei_path.exists() and cell_path.exists() and "--resegment" not in sys.argv:
        print("Reusing cached segmentation masks (pass --resegment to force a redo)")
        nuclei_masks = np.load(nuclei_path)
        cell_masks = np.load(cell_path)
    else:
        t0 = time.time()
        nuclei_masks = segment_nuclei_tiled(dapi, verbose=True)
        print(f"Nuclei tiling took {time.time()-t0:.1f}s, found {nuclei_masks.max()} nuclei")
        np.save(nuclei_path, nuclei_masks)

        t0 = time.time()
        _, cell_masks = segment_cells(dapi, gapdh, nuclei_masks=nuclei_masks)
        print(f"Watershed took {time.time()-t0:.1f}s")
        np.save(cell_path, cell_masks)

    artifact_mask = find_artifact_mask(x)
    print(f"Excluding {artifact_mask.sum()} non-biological artifact pixels from X channel")
    x_cell_masks = mask_out_artifacts(cell_masks, artifact_mask)

    df, info = filter_cells(nuclei_masks, cell_masks, gapdh, dcas=dcas if use_dcas_filter else None,
                             gapdh_thresh=gapdh_thresh, dcas_thresh=dcas_thresh,
                             min_cell_area=min_cell_area)
    df = add_mean_x(df, x_cell_masks, x)
    print("Thresholds:", info)
    print(f"Total: {len(df)}  real_cell: {df['real_cell'].sum()}  "
          f"final kept: {df['keep'].sum()}")

    csv_path = out_dir / f"{stem}_quantification.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} (all {len(df)} candidate nuclei, for debugging/QC)")

    # quantification.xlsx: ONLY fully-qualifying cells (green in the QC image --
    # real_cell AND, where applicable, dcas_positive). Rows for inactive/overlap-
    # flagged/dCas-negative cells are dropped entirely, not just marked. One tab per
    # (construct, ID, slide, slice) GROUP, merging all pictures of that same
    # physical section together -- per user, 2026-07-14. "pic" column preserves
    # which picture each row came from.
    keep_cols = ["label", "area", "cell_area", "solidity", "eccentricity", "mean_gapdh"]
    if "mean_dcas" in df.columns:
        keep_cols.append("mean_dcas")
    keep_cols.append("mean_X")
    qual_df = df.loc[df["keep"], keep_cols].reset_index(drop=True)
    qual_df.insert(0, "pic", pic_num)

    quant_xlsx_path = out_dir / "quantification.xlsx"
    combined = qual_df
    if quant_xlsx_path.exists():
        try:
            existing = pd.read_excel(quant_xlsx_path, sheet_name=group_sheet)
            existing = existing[existing["pic"].astype(str) != pic_num]  # replace this pic if rerun
            combined = pd.concat([existing, qual_df], ignore_index=True)
        except ValueError:
            pass  # sheet doesn't exist yet in this workbook
    mode = "a" if quant_xlsx_path.exists() else "w"
    kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(quant_xlsx_path, engine="openpyxl", mode=mode, **kwargs) as writer:
        combined.to_excel(writer, index=False, sheet_name=group_sheet)
    print(f"Saved {quant_xlsx_path} (sheet: {group_sheet}, {len(qual_df)} qualifying cells from "
          f"this picture, {len(combined)} total in the merged group)")

    # QC image -- DAPI=blue, GAPDH=green, dCas=red (per user, 2026-07-16)
    rgb = np.zeros((*dapi.shape, 3))
    rgb[..., 2] = to_display(dapi)
    rgb[..., 1] = to_display(gapdh) * 0.6
    rgb[..., 0] = to_display(dcas) * 0.8
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(rgb)
    for _, row in df.iterrows():
        lbl = int(row["label"])
        if row["keep"]:
            color = "lime"
        elif row["real_cell"]:
            color = "magenta"
        elif not row["active"]:
            color = "0.6"
        else:
            color = "orange"
        m = cell_masks == lbl
        if m.any():
            for c in find_contours(m.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.0)
    if use_dcas_filter:
        magenta_n = ((df["real_cell"]) & ~df["dcas_positive"]).sum()
        magenta_label = "dCas- real"
    else:
        magenta_n = 0
        magenta_label = "dCas- real (n/a, filter off)"
    ax.set_title(
        f"{filename}\nn={len(df)}: kept={df['keep'].sum()}(green) "
        f"{magenta_label}={magenta_n}(magenta) "
        f"inactive={(~df['active']).sum()}(gray) overlap={(~df['not_overlap']).sum()}(orange)"
    )
    ax.axis("off")
    ax.set_xlim(0, dapi.shape[1])
    ax.set_ylim(dapi.shape[0], 0)
    fig.tight_layout()
    qc_path = out_dir / f"{stem}_classification.png"
    fig.savefig(qc_path, dpi=150)
    print(f"Saved {qc_path}")

    # pooled histogram
    kept_labels = df.loc[df["keep"], "label"].tolist()
    counts, edges = pooled_histogram(x_cell_masks, x, kept_labels, bins=60)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="crimson", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("X channel pixel intensity")
    ax.set_ylabel("pixel count (log scale)")
    ax.set_title(f"{filename}\nPooled X-channel histogram, {len(kept_labels)} kept cells")
    fig.tight_layout()
    hist_path = out_dir / f"{stem}_x_pooled_histogram.png"
    fig.savefig(hist_path, dpi=130)
    print(f"Saved {hist_path}")

    # per-cell binned distributions -> shared distributions.xlsx, one tab per
    # (construct, ID, slide, slice) GROUP, merging all pictures of that section.
    # Bin range is FIXED (full uint16 range) rather than per-image max, so bins
    # line up exactly when merging cells from different pictures into one sheet.
    value_range = (0, 65535)
    bin_edges = np.linspace(value_range[0], value_range[1], N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dist_df = pd.DataFrame({
        "bin_start": bin_edges[:-1],
        "bin_end": bin_edges[1:],
        "bin_center": bin_centers,
    })
    for lbl in kept_labels:
        px = x[x_cell_masks == lbl]
        counts, _ = np.histogram(px, bins=bin_edges)
        dist_df[f"p{pic_num}_cell_{lbl}"] = counts

    xlsx_path = out_dir / "distributions.xlsx"
    combined = dist_df
    if xlsx_path.exists():
        try:
            existing = pd.read_excel(xlsx_path, sheet_name=group_sheet)
            other_pics = [c for c in existing.columns
                          if c not in ("bin_start", "bin_end", "bin_center")
                          and not c.startswith(f"p{pic_num}_cell_")]
            combined = pd.concat(
                [dist_df, existing[other_pics]], axis=1
            ) if other_pics else dist_df
        except ValueError:
            pass  # sheet doesn't exist yet in this workbook
    mode = "a" if xlsx_path.exists() else "w"
    kwargs = {"if_sheet_exists": "replace"} if mode == "a" else {}
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode=mode, **kwargs) as writer:
        combined.to_excel(writer, index=False, sheet_name=group_sheet)
    print(f"Saved {xlsx_path} (sheet: {group_sheet})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--resegment", action="store_true")
    parser.add_argument("--gapdh-thresh", type=float, default=GAPDH_ACTIVE_THRESHOLD)
    parser.add_argument("--dcas-thresh", type=float, default=DCAS_POSITIVE_THRESHOLD)
    parser.add_argument("--min-cell-area", type=float, default=MIN_CELL_AREA)
    args = parser.parse_args()
    main(args.filename, data_dir=args.data_dir, out_dir=args.out_dir,
         gapdh_thresh=args.gapdh_thresh, dcas_thresh=args.dcas_thresh,
         min_cell_area=args.min_cell_area)
