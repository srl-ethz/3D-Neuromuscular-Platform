# Axon–muscle overlap fraction

`axon_muscle_overlap.py` computes, for each sample, the percentage of the muscle area covered by axons, and plots the conditions side by side.

The overlap fraction is `100 × area(axon ∩ muscle) / area(muscle)`.

## Input

```
<input>/<condition>/<sample>/axon.tif          single-channel axon image
<input>/<condition>/<sample>/muscle_mask.tif   binary mask (nonzero = muscle)
```

The condition folder names are used as the x-axis labels.

## Usage

```bash
pip install numpy pandas scipy scikit-image tifffile matplotlib
python axon_muscle_overlap.py <input> --order Co-al Orth -o results
```

The pixel size is read from the resolution tags of `axon.tif` (override with `--pixel-size <um/px>`). `--order` sets the conditions from left to right (alphabetical by default). To redraw the plot from an existing CSV without re-running the segmentation:

```bash
python axon_muscle_overlap.py --csv results/overlap.csv --order Co-al Orth
```

## Output

- `overlap.csv`: one row per sample, with columns `condition`, `sample` and `overlap_pct`. Samples are numbered S1…Sn within each condition.
- `overlap_plot.png`: the dot plot.
