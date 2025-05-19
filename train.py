#!/usr/bin/env python3
"""
Trainer for the bead-tracking U-Net.

Run:
    python train.py              # real data  (uses config.yaml if present)
    python train.py --smoke      # 1-frame synthetic sanity check

Training is controlled via a fixed iteration count.
Default is 10k iterations. Checkpoints are saved every 50 iterations.
"""

import argparse, time, yaml
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import cycle

from scripts.logging_utils import setup_logger

from models.unet      import UNet
from scripts.dataset  import XRayBeadDataset
from scripts.heatmaps import generate_heatmap, softargmax_2d
from scripts.geometry import triangulate_torch, reproject_torch


logger = setup_logger(__name__)

# ---------------------------------------------------------------- helpers
def choose_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- training
def train(cfg, smoke=False):
    dev = choose_device()
    logger.info("Device: %s", dev)

    # ---------------- dataset --------------------------------------------
    if smoke:
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/labels").mkdir(parents=True, exist_ok=True)
        blank = np.zeros((128, 128), np.uint8)
        cv2.imwrite("data/raw/dummy_cam1.jpg", blank)
        cv2.imwrite("data/raw/dummy_cam2.jpg", blank)
        np.savetxt("data/labels/dummy_cam1.txt", [[64, 64, "B0"]], fmt="%s", delimiter=",")
        np.savetxt("data/labels/dummy_cam2.txt", [[64, 64, "B0"]], fmt="%s", delimiter=",")
        P = {"P": [[1,0,0,0],[0,1,0,0],[0,0,1,0]]}
        yaml.safe_dump(P, open("data/raw/cam1_dummy.yaml","w"))
        yaml.safe_dump(P, open("data/raw/cam2_dummy.yaml","w"))
        Path("data/train_list.txt").write_text(
            "raw/dummy_cam1.jpg,raw/dummy_cam2.jpg,"
            "labels/dummy_cam1.txt,labels/dummy_cam2.txt\n"
        )
        cfg["iters"] = 1
        ds = XRayBeadDataset("data/train_list.txt",
                             "data/raw/cam1_dummy.yaml",
                             "data/raw/cam2_dummy.yaml",
                             root="data")
    else:
        ds = XRayBeadDataset(cfg["split"],
                             cfg["cam1_yaml"],
                             cfg["cam2_yaml"],
                             root=cfg["root"])

    # do not stack batch elements so that lists remain untouched
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    collate_fn=lambda b: b[0])

    # ---------------- model / optimiser ---------------------------------
    net = UNet().to(dev)
    opt = optim.Adam(net.parameters(), lr=cfg["lr"])
    mse = nn.MSELoss();  lam = cfg["lambda"]

    # ---------------- iteration loop -----------------------------------
    total_iters = int(cfg.get("iters", len(dl)))
    dli = cycle(dl)
    t0 = time.time()
    for it in range(total_iters):
        smp = next(dli)

        # batch has been flattened by collate_fn; add batch dim
        img1 = smp["image1"].unsqueeze(0).to(dev)
        img2 = smp["image2"].unsqueeze(0).to(dev)
        logger.debug(
            "[iter %d] loaded images: img1 %s, img2 %s",
            it + 1,
            tuple(img1.shape),
            tuple(img2.shape),
        )

        kp1 = smp["kp1"]
        kp2 = smp["kp2"]
        logger.debug(
            "[iter %d] kp counts: cam1=%d cam2=%d",
            it + 1,
            len(kp1),
            len(kp2),
        )

        if len(kp1) == 0 or len(kp2) == 0:
            continue

        P1 = smp["P1"]
        P2 = smp["P2"]

        H, W = img1.shape[-2:]
        gt1 = generate_heatmap(kp1, H, W).to(dev)
        gt2 = generate_heatmap(kp2, H, W).to(dev)
        logger.debug(
            "[iter %d] gt1 max=%.3f gt2 max=%.3f",
            it + 1,
            gt1.max().item(),
            gt2.max().item(),
        )
        pred1 = net(img1);  pred2 = net(img2)
        logger.debug(
            "[iter %d] pred1 min=%.3f max=%.3f",
            it + 1,
            pred1.min().item(),
            pred1.max().item(),
        )
        loss_h = mse(pred1, gt1) + mse(pred2, gt2)

        x1, y1 = softargmax_2d(pred1)
        x2, y2 = softargmax_2d(pred2)
        x1 = x1[0]; y1 = y1[0]
        x2 = x2[0]; y2 = y2[0]
        XYZ = triangulate_torch(x1, y1, P1, x2, y2, P2)
        X, Y, Z = XYZ
        logger.debug(
            "[iter %d] triangulated point: (%.2f, %.2f, %.2f)",
            it + 1,
            X.item(),
            Y.item(),
            Z.item(),
        )
        rx1, ry1 = reproject_torch(XYZ, P1)
        rx2, ry2 = reproject_torch(XYZ, P2)
        loss_r = (rx1 - x1) ** 2 + (ry1 - y1) ** 2 + (rx2 - x2) ** 2 + (ry2 - y2) ** 2
        loss = loss_h + lam * loss_r
        logger.debug(
            "[iter %d] loss_h=%.4e loss_r=%.4e",
            it + 1,
            loss_h.item(),
            loss_r.item(),
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- visual debug every 200 iterations ----
        if (it + 1) % 200 == 0:
            Path("debug").mkdir(exist_ok=True)

            heat1 = cv2.applyColorMap(
                (pred1[0, 0].detach() * 255).byte().cpu().numpy(),
                cv2.COLORMAP_JET,
            )
            heat2 = cv2.applyColorMap(
                (pred2[0, 0].detach() * 255).byte().cpu().numpy(),
                cv2.COLORMAP_JET,
            )

            for x, y, _ in kp1:
                cv2.circle(heat1, (int(x), int(y)), 4, (0, 255, 0), -1)
            for x, y, _ in kp2:
                cv2.circle(heat2, (int(x), int(y)), 4, (0, 255, 0), -1)

            cv2.imwrite(f"debug/iter{it+1}_cam1.png", heat1)
            cv2.imwrite(f"debug/iter{it+1}_cam2.png", heat2)

        # ---- live progress ----
        if (it + 1) % 2 == 0:
            logger.info(
                "iter %d/%d loss=%.4e",
                it + 1,
                total_iters,
                loss.item(),
            )

        # ---- checkpoint every 50 iterations ----
        if (it + 1) % 50 == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(net.state_dict(), f"checkpoints/iter{it+1}.pt")
            logger.info(
                "\u2713 saved checkpoints/iter%d.pt  time=%.1fs",
                it + 1,
                time.time() - t0,
            )
            t0 = time.time()

    logger.info("Training finished.")


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke",  action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {
        "root":   "data",
        "split":  "data/train_list.txt",
        "cam1_yaml": "data/raw/cam1.yaml",
        "cam2_yaml": "data/raw/cam2.yaml",
        "iters": 10000,
        "lr":     5e-5,      # lower LR to escape plateau
        "lambda": 0.02,      # smaller reprojection weight
    }
    train(cfg, smoke=args.smoke)
