# 3D Bead Tracker

This project contains a small experimental pipeline for detecting radiopaque beads in stereo X-ray images. The core components are:

- **UNet** model implemented in `models/unet.py`.
- **Dataset utilities** under the `scripts/` directory.
- **Training entry point** in `train.py` (now iteration-based).

The repository includes a tiny example dataset in `data/` and various helper
scripts for creating calibration files and visualising predictions.

Training now runs for a fixed number of iterations (default 10k) with verbose
debugging information printed every step. Checkpoints are saved every 50
iterations.

## Setup

1. **Install dependencies** (Python 3.8+):

   ```bash
   pip install -r requirements.txt
   ```

   The requirements file lists `torch`, `torchvision`,
   `opencv-python`, `pyyaml`, `numpy` and `pandas`.

2. **Create camera calibration files**.  Edit the matrices in
   `scripts/build_yaml_from_calibration.py` with your calibration numbers and
   run:

   ```bash
   python scripts/build_yaml_from_calibration.py
   ```

   This writes `data/raw/cam1.yaml` and `data/raw/cam2.yaml`.

3. **Run training** using the default parameters or a custom `config.yaml`:

   ```bash
   python train.py              # uses config.yaml if present
   python train.py --smoke      # tiny synthetic sanity check
   python train.py --config cfg.yaml --lr 3e-4 --lambda 0.02 --focal-gamma 2
   ```

Example `config.yaml` snippet:

```yaml
root: data
split: data/train_list.txt
cam1_yaml: data/raw/cam1.yaml
cam2_yaml: data/raw/cam2.yaml
iters: 10000
lr: 1e-4
lambda: 0.02
focal_gamma: 2.0
focal_alpha: 0.25
```

TensorBoard logs are written to `runs/`. Launch with:

```bash
tensorboard --logdir runs
```

![TensorBoard screenshot](docs/tb_placeholder.png)

## Troubleshooting

The heatmap generation utilities create peaks with a maximum value of `1.0`. If
debug output mentions `gt1 max=1.0` this simply reflects that behaviour and is
expected.

When predictions appear clustered on one side of the images, visualise the
output using `scripts/viz_pred_two_cam.py` to inspect the points from both
cameras. This can help reveal issues with misalignment.

In such cases also double‑check your camera calibration YAML files to confirm
the extrinsic matrices are correct.

## Inference

Run a trained model on new images or videos using `scripts/inference.py`.
The script expects the following arguments:

1. `--checkpoint` – path to a `.pt` weight file.
2. `--cam1_yaml` and `--cam2_yaml` – camera calibration files. These default
   to `data/raw/cam1.yaml` and `data/raw/cam2.yaml` if not given.
3. Either `--pairs LIST.txt` containing comma‑separated image pairs, or
   `--video1` and `--video2` for two synchronised videos.
4. `--out` – destination CSV file.

Use `--cam1_yaml` and `--cam2_yaml` to provide custom calibration files if they
are stored elsewhere.

Example command for a list of images:

```bash
python scripts/inference.py --checkpoint checkpoints/epoch100.pt \
       --pairs data/test_list.txt \
       --out results.csv
```

Running on videos works in the same way:

```bash
python scripts/inference.py --checkpoint checkpoints/epoch100.pt \
       --video1 cam1.mov --video2 cam2.mov \
       --out results.csv
```

The output CSV contains one row per frame with the 3‑D coordinates of each bead
in the order defined by `scripts/heatmaps_multi.py`:

```python
IDS = [
    'RAD1', 'RAD2', 'RAD3',
    'MCIII1', 'MCIII2', 'MCIII3',
]  # order must stay fixed
```

XMA expects this ordering, so the generated CSV can be imported directly.

