# Microscopy-Image-Analysis-Pipeline
Analysis pipeline to identify individual cells in brain sections of adult mice from fluorescence confocal microscopy images with multiple Z-stacks and three-channel HCR probes. This pipeline uses Cellpose (v4, GPU-accelerated) to segment nuclei via DAPI (blue), then expands each nucleus into a whole-cell boundary using a watershed step guided by AlexaFluor-488 signal (GAPDH, green) and filters for GAPDH-active and non-overlapping cells. It then screens for high AAV transduction via the AlexaFluor-546 channel (dCas probes, purple), and calculates mean pixel signal of the target mRNA in the remaining AlexaFluor-647 channel (red). Image analysis code was developed with assistance from Claude Code (Anthropic).

# Image analysis pipeline

Segmentation, filtering, and quantification code for CRISPRa/dCas delivery-screen
microscopy (Zeiss LSM900 + Airyscan, 4-channel FISH-style .czi images: DAPI, GAPDH,
dCas, target transcript). Given one raw `.czi` file, the pipeline segments nuclei and
whole cells, filters to real/active/singlet cells, and quantifies mean target-channel
intensity per cell.

## Files

- **`load_czi.py`** — Reads a `.czi` file and returns its DAPI/GAPDH/dCas/target-transcript
  channels as 2D arrays (max-intensity projected across z, first repeat scan only).
- **`segment.py`** — Nuclei segmentation via Cellpose, run per-tile (800×800 px) and
  stitched with a global dedup step, `flow_threshold=0.6`, minimum nucleus area 3,000 px².
  Whole-cell boundaries via marker-controlled watershed, seeded from nuclei, over a
  grayscale-closed GAPDH channel.
- **`filter_cells.py`** — Per-cell feature table (GAPDH/dCas mean intensity, nucleus/cell
  area, solidity, eccentricity) plus the filter chain: GAPDH-active (≥1,000), non-overlapping
  nucleus, whole-cell mask ≥8,000 px², and (for CRISPRa/IFFL constructs) dCas-positive
  (≥6,000). Thresholds are fixed absolute values, not per-image Otsu.
- **`quantify.py`** — Masks out saturated non-biological artifacts in the target channel
  (≥99.5th percentile intensity, ≥35 px equivalent diameter) and computes mean target-channel
  intensity per retained cell.
- **`process_image.py`** — End-to-end driver: load → segment (or reuse cached masks) →
  filter → quantify → export per-cell CSV, a classification overlay PNG, and a pooled
  histogram. Auto-detects GFP-construct filenames and skips the dCas filter for them.

## Usage

```bash
python process_image.py "<filename.czi>" --data-dir "<path/to/raw czi files>" --out-dir "<path/for/output>"
```

Segmentation masks are cached per image (`<stem>_nuclei_masks.npy` / `<stem>_cell_masks.npy`
in `--out-dir`) and reused on rerun; pass `--resegment` to force a redo. Thresholds can be
overridden with `--gapdh-thresh`, `--dcas-thresh`, `--min-cell-area`.

## Dependencies

`cellpose>=4`, `numpy`, `pandas`, `scikit-image`, `scipy`, `matplotlib`, plus a `.czi`
reader (`aicsimageio` or equivalent, used inside `load_czi.py`). Cellpose segmentation
uses GPU acceleration (Apple MPS / CUDA) if available.

## Notes

- All thresholds (`GAPDH_ACTIVE_THRESHOLD`, `DCAS_POSITIVE_THRESHOLD`, `MIN_CELL_AREA` in
  `filter_cells.py`; `MIN_NUCLEUS_AREA`, `flow_threshold` in `segment.py`) were set by
  iterative comparison against manual expert annotation on a held-out reference image,
  not tuned per-dataset.
- This folder is a static copy for submission/reproducibility purposes. It is
  self-contained (imports resolve locally) and does not depend on the rest of the
  original project repository.
