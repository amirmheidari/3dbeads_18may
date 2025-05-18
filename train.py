#!/usr/bin/env python3
"""
Trainer for the bead-tracking U-Net.

Run:
    python train.py              # real data  (uses config.yaml if present)
    python train.py --smoke      # 1-frame synthetic sanity check
"""

import argparse, time, yaml
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet      import UNet
from scripts.dataset  import XRayBeadDataset
from scripts.heatmaps import generate_heatmap
from scripts.geometry import triangulate, reproject


# ---------------------------------------------------------------- helpers
def choose_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def peak_xy(hm):
    idx = torch.argmax(hm).item()
    H, W = hm.shape[-2:]
    y, x = divmod(idx, W)
    return x, y


# ---------------------------------------------------------------- training
def train(cfg, smoke=False):
    dev = choose_device()
    print("Device:", dev)

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
        cfg["epochs"] = 1
        ds = XRayBeadDataset("data/train_list.txt",
                             "data/raw/cam1_dummy.yaml",
                             "data/raw/cam2_dummy.yaml",
                             root="data")
    else:
        ds = XRayBeadDataset(cfg["split"],
                             cfg["cam1_yaml"],
                             cfg["cam2_yaml"],
                             root=cfg["root"])

    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # ---------------- model / optimiser ---------------------------------
    net = UNet().to(dev)
    opt = optim.Adam(net.parameters(), lr=cfg["lr"])
    mse = nn.MSELoss();  lam = cfg["lambda"]

    # ---------------- epoch loop ----------------------------------------
    for ep in range(int(cfg["epochs"])):
        t0 = time.time()
        for idx, smp in enumerate(dl):
            img1 = smp["image1"].to(dev)
            img2 = smp["image2"].to(dev)

            kp1_list = smp["kp1"]
            kp2_list = smp["kp2"]

            if (not kp1_list or not kp2_list or
                len(kp1_list[0]) == 0 or len(kp2_list[0]) == 0):
                continue

            kp1 = kp1_list[0];   kp2 = kp2_list[0]
            P1  = smp["P1"][0];  P2 = smp["P2"][0]

            H, W = img1.shape[-2:]
            gt1 = generate_heatmap(kp1, H, W).to(dev)
            gt2 = generate_heatmap(kp2, H, W).to(dev)
            pred1 = net(img1);  pred2 = net(img2)
            loss_h = mse(pred1, gt1) + mse(pred2, gt2)

            x1, y1 = peak_xy(pred1[0,0]);  x2, y2 = peak_xy(pred2[0,0])
            X,Y,Z  = triangulate(x1,y1,P1, x2,y2,P2)
            rx1,ry1 = reproject((X,Y,Z), P1);  rx2,ry2 = reproject((X,Y,Z), P2)
            reproj  = (rx1-x1)**2 + (ry1-y1)**2 + (rx2-x2)**2 + (ry2-y2)**2
            reproj  = 0.0 if not np.isfinite(reproj) else reproj
            loss_r  = torch.as_tensor(reproj, device=dev, dtype=pred1.dtype)
            loss    = loss_h + lam*loss_r

            opt.zero_grad(); loss.backward(); opt.step()

            # ---- visual debug every 200 batches ----
            if (idx + 1) % 200 == 0:
                Path("debug").mkdir(exist_ok=True)
                vis = (pred1[0,0].detach().cpu().numpy()*255).astype("uint8")
                cv2.imwrite(f"debug/ep{ep+1}_it{idx+1}.png", vis)

            # ---- live progress ----
            print(f"epoch {ep+1}/{cfg['epochs']}  "
                  f"iter {idx+1}/{len(dl)}  "
                  f"loss={loss.item():.4e}", end="\r")

        print()  # newline
        print(f"Epoch {ep+1}/{cfg['epochs']}  loss={loss.item():.4e}  "
              f"time={time.time()-t0:.1f}s")

        # ---------- checkpoint ----------
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(net.state_dict(), f"checkpoints/epoch{ep+1}.pt")
        print(f"✓ saved checkpoints/epoch{ep+1}.pt")

    print("Training finished.")


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
        "epochs": 5,
        "lr":     5e-5,      # lower LR to escape plateau
        "lambda": 0.02,      # smaller reprojection weight
    }
    train(cfg, smoke=args.smoke)