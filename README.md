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

