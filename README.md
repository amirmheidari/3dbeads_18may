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
   `opencv-python`, `pyyaml` and `numpy`.

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
   ```

Example `config.yaml` snippet:

```yaml
root: data
split: data/train_list.txt
cam1_yaml: data/raw/cam1.yaml
cam2_yaml: data/raw/cam2.yaml
iters: 10000
lr: 5e-5
lambda: 0.02
```

## Troubleshooting

The heatmap generation utilities create peaks with a maximum value of `1.0`. If
debug output mentions `gt1 max=1.0` this simply reflects that behaviour and is
expected.

When predictions appear clustered on one side of the images, visualise the
output using `scripts/viz_pred_two_cam.py` to inspect the points from both
cameras. This can help reveal issues with misalignment.

In such cases also double‑check your camera calibration YAML files to confirm
the extrinsic matrices are correct.

