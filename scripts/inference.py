#!/usr/bin/env python3
"""Infer bead locations from stereo images or videos."""

import argparse
import csv
from pathlib import Path

import cv2
import torch

from models.unet import UNet
from scripts.heatmaps import softargmax_2d
from scripts.heatmaps_multi import IDS
from scripts.logging_utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    img = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
    return img


# ---------------------------------------------------------------------------
def run_model(net: UNet, img: torch.Tensor, dev: torch.device):
    with torch.no_grad():
        pred = net(img.unsqueeze(0).to(dev))[0]
    x, y = softargmax_2d(pred.unsqueeze(0))
    return x[0].cpu(), y[0].cpu(), pred.cpu()


# ---------------------------------------------------------------------------
def draw_points(gray: torch.Tensor, pts):
    vis = cv2.cvtColor((gray.squeeze().cpu().numpy() * 255).astype("uint8"),
                       cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
    return vis


# ---------------------------------------------------------------------------
def process_list(list_file: Path, root: Path):
    pairs = []
    with open(list_file, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split(",")
            if len(parts) < 2:
                continue
            img1 = root / parts[0]
            img2 = root / parts[1]
            pairs.append((img1, img2))
    return pairs


# ---------------------------------------------------------------------------
def process_videos(v1: Path, v2: Path):
    cap1 = cv2.VideoCapture(str(v1))
    cap2 = cv2.VideoCapture(str(v2))
    if not cap1.isOpened() or not cap2.isOpened():
        raise FileNotFoundError("Video open failed")
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not (ret1 and ret2):
            break
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        yield torch.from_numpy(g1.astype("float32") / 255.0).unsqueeze(0), \
            torch.from_numpy(g2.astype("float32") / 255.0).unsqueeze(0)
    cap1.release()
    cap2.release()


# ---------------------------------------------------------------------------
def main(args):
    dev = choose_device()
    net = UNet(out_channels=len(IDS))
    net.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    net.to(dev).eval()
    logger.info("Loaded %s", args.checkpoint)

    # gather input frames
    if args.list:
        pairs = process_list(Path(args.list), Path(args.root))
        def frame_gen():
            for img1, img2 in pairs:
                yield load_image(img1), load_image(img2)
    else:
        def frame_gen():
            yield from process_videos(Path(args.cam1), Path(args.cam2))

    cols = []
    for bead in IDS:
        cols.extend([f"{bead}_cam1_X", f"{bead}_cam1_Y",
                     f"{bead}_cam2_X", f"{bead}_cam2_Y"])

    out_rows = []
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img1, img2) in enumerate(frame_gen()):
        x1, y1, pred1 = run_model(net, img1, dev)
        x2, y2, pred2 = run_model(net, img2, dev)

        row = []
        for k in range(len(IDS)):
            row.extend([
                float(x1[k].item()),
                float(y1[k].item()),
                float(x2[k].item()),
                float(y2[k].item()),
            ])
        out_rows.append(row)

        if debug_dir:
            vis1 = draw_points(img1, zip(x1, y1))
            vis2 = draw_points(img2, zip(x2, y2))
            cv2.imwrite(str(debug_dir / f"frame{idx:04d}_cam1.png"), vis1)
            cv2.imwrite(str(debug_dir / f"frame{idx:04d}_cam2.png"), vis2)
            heat1 = cv2.applyColorMap((pred1[0] * 255).byte().numpy(),
                                     cv2.COLORMAP_JET)
            heat2 = cv2.applyColorMap((pred2[0] * 255).byte().numpy(),
                                     cv2.COLORMAP_JET)
            cv2.imwrite(str(debug_dir / f"frame{idx:04d}_cam1_heat.png"), heat1)
            cv2.imwrite(str(debug_dir / f"frame{idx:04d}_cam2_heat.png"), heat2)

    # write CSV
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(out_rows)
    logger.info("Wrote %s with %d frames", args.out_csv, len(out_rows))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run inference on a trained model")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--list", help="Frame list like train_list.txt")
    ap.add_argument("--cam1", help="Camera-1 video when not using --list")
    ap.add_argument("--cam2", help="Camera-2 video when not using --list")
    ap.add_argument("--root", default=".", help="Root directory for list paths")
    ap.add_argument("--out-csv", default="pred.csv", dest="out_csv",
                    help="CSV output path")
    ap.add_argument("--debug-dir", help="Write PNGs to this directory")
    args = ap.parse_args()

    if not args.list and not (args.cam1 and args.cam2):
        ap.error("Specify --list FILE or both --cam1 and --cam2 videos")

    main(args)
