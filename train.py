#!/usr/bin/env python3
"""
Trainer for the bead-tracking U-Net.

Run:
    python train.py              # real data  (uses config.yaml if present)
    python train.py --smoke      # 1-frame synthetic sanity check

Training is controlled via a fixed iteration count.
Default is 10k iterations. Checkpoints are saved every 50 iterations.
"""

import argparse
import logging
import math
import time
import yaml
from pathlib import Path
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from itertools import cycle

from scripts.logging_utils import setup_logger

from models.unet      import UNet
from scripts.dataset  import XRayBeadDataset
from scripts.heatmaps_multi import generate_multichannel_heatmaps, IDS
from scripts.heatmaps import softargmax_2d

K = len(IDS)
from scripts.geometry import triangulate_torch, reproject_torch


class FocalLoss(nn.Module):
    """Binary focal loss for heatmap regression."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)
        loss_pos = -self.alpha * (1 - p) ** self.gamma * targets * torch.log(p)
        loss_neg = -(1 - self.alpha) * p ** self.gamma * (1 - targets) * torch.log(1 - p)
        return (loss_pos + loss_neg).mean()


logger = setup_logger(__name__)

# ---------------------------------------------------------------- helpers
def choose_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- training
def train(cfg, smoke: bool = False, iters_override: int | None = None):
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
        cfg["iters"] = iters_override or 1
        ds = XRayBeadDataset("data/train_list.txt",
                             "data/raw/cam1_dummy.yaml",
                             "data/raw/cam2_dummy.yaml",
                             root="data")
    else:
        ds = XRayBeadDataset(cfg["split"],
                             cfg["cam1_yaml"],
                             cfg["cam2_yaml"],
                             root=cfg["root"])

    writer = SummaryWriter()

    # do not stack batch elements so that lists remain untouched
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    collate_fn=lambda b: b[0])

    # ---------------- model / optimiser ---------------------------------
    net = UNet(in_ch=3, out_channels=K).to(dev)
    opt = optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"])
    criterion = FocalLoss(cfg["focal_alpha"], cfg["focal_gamma"])
    lam = cfg["lambda"]
    beta = cfg.get("beta", 0.05)

    # ---------------- iteration loop -----------------------------------
    total_iters = int(cfg.get("iters", len(dl)))
    warmup = min(500, total_iters // 4)
    ramp = min(2000, total_iters - warmup)

    def lam_schedule(i: int) -> float:
        if i < warmup:
            return 0.0
        if i < warmup + ramp:
            p = (i - warmup) / ramp
            return lam * 0.5 * (1 - math.cos(math.pi * p))
        return lam
    dli = cycle(dl)
    t0 = time.time()
    for it in range(total_iters):
        smp = next(dli)

        # batch has been flattened by collate_fn; add batch dim
        img1 = smp["image1"].unsqueeze(0).to(dev)
        img2 = smp["image2"].unsqueeze(0).to(dev)
        B, _, H, W = img1.shape
        xx = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, 1, H, W)
        img1 = torch.cat([img1, xx, yy], 1)
        img2 = torch.cat([img2, xx, yy], 1)
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
        gt1_dict = {b: (x, y) for x, y, b in kp1}
        gt2_dict = {b: (x, y) for x, y, b in kp2}
        for x, y, bead in kp1:
            logger.debug(
                "[iter %d] cam1 gt %-6s (%.1f, %.1f)",
                it + 1,
                bead,
                x,
                y,
            )
        for x, y, bead in kp2:
            logger.debug(
                "[iter %d] cam2 gt %-6s (%.1f, %.1f)",
                it + 1,
                bead,
                x,
                y,
            )

        if len(kp1) == 0 or len(kp2) == 0:
            continue

        P1 = smp["P1"]
        P2 = smp["P2"]

        H, W = img1.shape[-2:]
        gt1 = generate_multichannel_heatmaps(kp1, H, W).to(dev)
        gt2 = generate_multichannel_heatmaps(kp2, H, W).to(dev)
        logger.debug(
            "[iter %d] gt1 max=%.3f gt2 max=%.3f",
            it + 1,
            gt1.max().item(),
            gt2.max().item(),
        )
        pred1 = net(img1)[0]
        pred2 = net(img2)[0]
        logger.debug(
            "[iter %d] pred1 min=%.3f max=%.3f",
            it + 1,
            pred1.min().item(),
            pred1.max().item(),
        )
        loss_h = criterion(pred1, gt1) + criterion(pred2, gt2)
        prob1 = torch.sigmoid(pred1)
        prob2 = torch.sigmoid(pred2)
        loss_sep = ((prob1.sum(0, keepdim=True) - prob1) * prob1).mean() + \
                   ((prob2.sum(0, keepdim=True) - prob2) * prob2).mean()

        x1, y1 = softargmax_2d(pred1.unsqueeze(0))
        x2, y2 = softargmax_2d(pred2.unsqueeze(0))
        x1 = x1[0]
        y1 = y1[0]
        x2 = x2[0]
        y2 = y2[0]
        for k, bead in enumerate(IDS):
            g1 = gt1_dict.get(bead)
            g2 = gt2_dict.get(bead)
            logger.debug(
                "[iter %d] bead %-6s pred cam1=(%.1f, %.1f) gt1=%s cam2=(%.1f, %.1f) gt2=%s",
                it + 1,
                bead,
                x1[k].item(),
                y1[k].item(),
                g1,
                x2[k].item(),
                y2[k].item(),
                g2,
            )
        loss_r = 0
        for k, bead in enumerate(IDS):
            XYZ = triangulate_torch(x1[k], y1[k], P1, x2[k], y2[k], P2)
            X, Y, Z = XYZ
            logger.debug(
                "[iter %d] bead %-6s XYZ=(%.2f, %.2f, %.2f)",
                it + 1,
                bead,
                X.item(),
                Y.item(),
                Z.item(),
            )
            rx1, ry1 = reproject_torch(XYZ, P1)
            rx2, ry2 = reproject_torch(XYZ, P2)
            logger.debug(
                "[iter %d] bead %-6s reproj1=(%.1f, %.1f) reproj2=(%.1f, %.1f)",
                it + 1,
                bead,
                rx1.item(),
                ry1.item(),
                rx2.item(),
                ry2.item(),
            )
            loss_r = loss_r + (rx1 - x1[k]) ** 2 + (ry1 - y1[k]) ** 2 + (rx2 - x2[k]) ** 2 + (ry2 - y2[k]) ** 2
        lam_t = lam_schedule(it)
        loss = loss_h + lam_t * loss_r + beta * loss_sep
        logger.debug(
            "[iter %d] loss_h=%.4e loss_r=%.4e loss_sep=%.4e lam=%.4e",
            it + 1,
            loss_h.item(),
            loss_r.item(),
            loss_sep.item(),
            lam_t,
        )

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        writer.add_scalar("loss/h", loss_h.item(), it + 1)
        writer.add_scalar("loss/r", loss_r.item(), it + 1)
        writer.add_scalar("loss/sep", loss_sep.item(), it + 1)

        # ---- visual debug every 200 iterations ----
        if (it + 1) % 200 == 0:
            Path("debug").mkdir(exist_ok=True)
            img_dbg1 = cv2.cvtColor((smp["image1"][0].cpu().numpy()*255).astype("uint8"), cv2.COLOR_GRAY2BGR)
            img_dbg2 = cv2.cvtColor((smp["image2"][0].cpu().numpy()*255).astype("uint8"), cv2.COLOR_GRAY2BGR)
            tb_imgs = []
            for k, bead in enumerate(IDS):
                h1 = cv2.applyColorMap((prob1[k].detach()*255).byte().cpu().numpy(), cv2.COLORMAP_JET)
                h2 = cv2.applyColorMap((prob2[k].detach()*255).byte().cpu().numpy(), cv2.COLORMAP_JET)
                for x, y, _ in kp1:
                    cv2.circle(h1, (int(x), int(y)), 4, (0, 255, 0), -1)
                for x, y, _ in kp2:
                    cv2.circle(h2, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.imwrite(f"debug/iter{it+1}_cam1_{bead}.png", h1)
                cv2.imwrite(f"debug/iter{it+1}_cam2_{bead}.png", h2)
                ov1 = cv2.addWeighted(img_dbg1, 0.5, h1, 0.5, 0)
                ov2 = cv2.addWeighted(img_dbg2, 0.5, h2, 0.5, 0)
                tb_imgs.extend([ov1[..., ::-1], ov2[..., ::-1]])
            grid = make_grid(
                torch.from_numpy(np.stack(tb_imgs).astype("float32") / 255)
                .permute(0, 3, 1, 2),
                nrow=len(IDS) * 2,
            )
            writer.add_image("pred", grid, it + 1)

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
    writer.close()


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--lambda", dest="lambda_", type=float)
    ap.add_argument("--focal-gamma", type=float)
    ap.add_argument("--focal-alpha", type=float)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {
        "root": "data",
        "split": "data/train_list.txt",
        "cam1_yaml": "data/raw/cam1.yaml",
        "cam2_yaml": "data/raw/cam2.yaml",
        "iters": 10000,
        "lr": 1e-4,
        "lambda": 0.02,
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "beta": 0.05,
    }
    for key, val in [("lr", args.lr), ("lambda", args.lambda_),
                     ("focal_gamma", args.focal_gamma),
                     ("focal_alpha", args.focal_alpha),
                     ("iters", args.iters)]:
        if val is not None:
            cfg[key] = val

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    from scripts import dataset as dataset_mod
    dataset_mod.logger.setLevel(log_level)
    logger.setLevel(log_level)
    import scripts.logging_utils as logging_utils
    logging_utils.DEFAULT_LEVEL = log_level

    train(cfg, smoke=args.smoke, iters_override=args.iters)
