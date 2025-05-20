#!/usr/bin/env python3
"""Run a trained model on image pairs or videos and output 3-D bead positions."""

import argparse
import csv
import yaml
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.logging_utils import setup_logger
from models.unet import UNet
from scripts.heatmaps_multi import IDS
from scripts.heatmaps import softargmax_2d
from scripts.geometry import triangulate

logger = setup_logger(__name__)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_P(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        P = np.asarray(yaml.safe_load(f)["P"], np.float32)
    if P.shape != (3, 4):
        raise ValueError(f"{path}: projection must be 3x4, got {P.shape}")
    return P


# -----------------------------------------------------------------------------

def run_on_images(pairs, net, dev, P1, P2):
    results = []
    for img1_fp, img2_fp in pairs:
        logger.info("Processing %s %s", img1_fp, img2_fp)
        img1 = cv2.imread(str(img1_fp), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_fp), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            logger.error("Missing %s or %s", img1_fp, img2_fp)
            raise FileNotFoundError(f"Missing {img1_fp} or {img2_fp}")
        img1_t = torch.from_numpy(img1.astype("float32") / 255).unsqueeze(0).unsqueeze(0)
        img2_t = torch.from_numpy(img2.astype("float32") / 255).unsqueeze(0).unsqueeze(0)
        B, _, H, W = img1_t.shape
        xx = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)
        img1_t = torch.cat([img1_t, xx, yy], 1)
        img2_t = torch.cat([img2_t, xx, yy], 1)
        with torch.no_grad():
            hm1 = net(img1_t.to(dev))[0]
            hm2 = net(img2_t.to(dev))[0]
        x1, y1 = softargmax_2d(hm1.unsqueeze(0))
        x2, y2 = softargmax_2d(hm2.unsqueeze(0))
        x1 = x1[0].cpu().numpy()
        y1 = y1[0].cpu().numpy()
        x2 = x2[0].cpu().numpy()
        y2 = y2[0].cpu().numpy()
        row = []
        for k in range(len(IDS)):
            X, Y, Z = triangulate(float(x1[k]), float(y1[k]), P1,
                                  float(x2[k]), float(y2[k]), P2)
            row.extend([X, Y, Z])
        results.append(row)
    return results


def run_on_videos(v1_path: Path, v2_path: Path, net, dev, P1, P2):
    cap1 = cv2.VideoCapture(str(v1_path))
    cap2 = cv2.VideoCapture(str(v2_path))
    if not cap1.isOpened() or not cap2.isOpened():
        raise FileNotFoundError("Could not open videos")
    results = []
    idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        logger.info("Frame %d", idx)
        if frame1.ndim == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if frame2.ndim == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        img1_t = torch.from_numpy(frame1.astype("float32") / 255).unsqueeze(0).unsqueeze(0)
        img2_t = torch.from_numpy(frame2.astype("float32") / 255).unsqueeze(0).unsqueeze(0)
        B, _, H, W = img1_t.shape
        xx = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(B, 1, H, W)
        img1_t = torch.cat([img1_t, xx, yy], 1)
        img2_t = torch.cat([img2_t, xx, yy], 1)
        with torch.no_grad():
            hm1 = net(img1_t.to(dev))[0]
            hm2 = net(img2_t.to(dev))[0]
        x1, y1 = softargmax_2d(hm1.unsqueeze(0))
        x2, y2 = softargmax_2d(hm2.unsqueeze(0))
        x1 = x1[0].cpu().numpy()
        y1 = y1[0].cpu().numpy()
        x2 = x2[0].cpu().numpy()
        y2 = y2[0].cpu().numpy()
        row = []
        for k in range(len(IDS)):
            X, Y, Z = triangulate(float(x1[k]), float(y1[k]), P1,
                                  float(x2[k]), float(y2[k]), P2)
            row.extend([X, Y, Z])
        results.append(row)
        idx += 1
    cap1.release()
    cap2.release()
    return results


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="3-D bead inference")
    ap.add_argument("--checkpoint", required=True, type=Path)
    default_base = Path(__file__).resolve().parent.parent / "data" / "raw"
    ap.add_argument(
        "--cam1_yaml",
        type=Path,
        default=default_base / "cam1.yaml",
        help="Camera 1 calibration YAML (default: data/raw/cam1.yaml)",
    )
    ap.add_argument(
        "--cam2_yaml",
        type=Path,
        default=default_base / "cam2.yaml",
        help="Camera 2 calibration YAML (default: data/raw/cam2.yaml)",
    )
    ap.add_argument("--pairs", type=Path, help="Text file with image pairs")
    ap.add_argument("--video1", type=Path, help="Camera 1 video")
    ap.add_argument("--video2", type=Path, help="Camera 2 video")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.pairs and (args.video1 or args.video2):
        ap.error("Use either --pairs or --video1/--video2")
    if (args.video1 and not args.video2) or (args.video2 and not args.video1):
        ap.error("--video1 and --video2 must be given together")
    if not args.pairs and not (args.video1 and args.video2):
        ap.error("Specify --pairs or --video1/--video2")

    dev = choose_device()
    logger.info("Device: %s", dev)

    net = UNet(in_ch=3, out_channels=len(IDS))
    net.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    net.to(dev).eval()
    logger.info("\u2713 loaded %s", args.checkpoint)

    P1 = load_P(args.cam1_yaml)
    P2 = load_P(args.cam2_yaml)

    if args.pairs:
        with open(args.pairs, "r", encoding="utf-8") as f:
            pairs = [tuple(ln.strip().split(",")[:2]) for ln in f if ln.strip()]
        rows = run_on_images(pairs, net, dev, P1, P2)
    else:
        rows = run_on_videos(args.video1, args.video2, net, dev, P1, P2)

    header = [f"{bid}_{ax}" for bid in IDS for ax in ("X", "Y", "Z")]
    with open(args.out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("\u2713 wrote %s", args.out)


if __name__ == "__main__":
    main()
