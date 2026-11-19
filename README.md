# Player Location Coordinate Fusion

Player positions on the pitch are estimated from YOLO26n detections and MobileSAM masks. The box and mask ground points are blended separately for each image axis, with weights that depend on the box size. Results are reported on Spiideo SoccerNet SynLoc.

## Contents

- `models/yolo26n_fullhd/yolo26n_fullhd-2/`: trained detector (`weights/best.pt`) and training logs
- `output/`: cached detections, masks and keypoints, and all evaluation results reported in the paper
- `01_*`, `02_*`, `03_*`: cache detections, poses and masks
- `evaluate_*`: evaluate localization methods
- `analyze_*`: error analysis
- `bootstrap_synloc.py`: bootstrap confidence intervals
- `generate_synloc_fig*.py`: paper figures

"Hybrid A" in file names is the method called "Hybrid" in the paper.

## Data

Download SynLoc with [sskit](https://github.com/Spiideo/sskit) and put it in `data/spiideo/` (or pass its location with `--dataset-root`). The dataset is not included in this repository.

## Reproduce

1. Cache detections, poses and masks: `01_cache_synloc_yolo.py`, `02_cache_synloc_pose_from_yolo.py`, `03_cache_synloc_sam_from_yolo.py`
2. Evaluate the methods: `evaluate_synloc_sam_size_adaptive.py`, `evaluate_synloc_sam_fusion_strategies.py`, `evaluate_synloc_sam_continuous_x.py`
3. Compute statistics and figures: `bootstrap_synloc.py`, `generate_synloc_fig*.py`

Run any script with `--help` to see its options. Steps 2 and 3 can be run directly from the cached results in `output/`, without a GPU.

## License

AGPL-3.0