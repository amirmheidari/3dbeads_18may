#!/usr/bin/env python3
# =============================================================
#  TWO-CAMERA VISUALISER
# =============================================================
"""
usage:
    python scripts/viz_pred_two_cam.py  checkpoints/epochN.pt  [row]
Writes 4 PNGs to ./debug/
"""

import sys, cv2, numpy as np, torch
from pathlib import Path

from scripts.logging_utils import setup_logger

logger = setup_logger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))   # project root

from models.unet      import UNet
from scripts.dataset  import XRayBeadDataset
from scripts.geometry import triangulate, reproject

# -------- helpers -------------------------------------------------
def peak_xy(hm: torch.Tensor):
    idx = int(hm.argmax()); H, W = hm.shape; y, x = divmod(idx, W); return x, y

def as_list(arr):
    if isinstance(arr, list):      # already list[tuple]
        return arr
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size >= 2:
        return [tuple(float(v) for v in arr[:2]) +
                (arr[2] if arr.size > 2 else "",)]
    out = []
    for row in arr:
        out.append(tuple(float(v) for v in row[:2]) +
                    (row[2] if len(row) > 2 else "",))
    return out

def draw(gray, gt, pred, rep):
    if isinstance(gray, torch.Tensor):
        gray = gray.cpu().numpy()
    vis = cv2.cvtColor((gray * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)
    for x, y, _ in gt:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)   # GT
    cv2.circle(vis, pred, 4, (0, 0, 255), -1)                   # peak
    cv2.circle(vis, rep,  4, (255, 255, 0), -1)                 # reproj
    return vis

# -------- CLI -----------------------------------------------------
if len(sys.argv) < 2:
    sys.exit("usage: viz_pred_two_cam.py  checkpoint.pt  [row]")
ckpt = Path(sys.argv[1]); row = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev  = "mps" if torch.backends.mps.is_available() else "cpu"

# -------- model ---------------------------------------------------
net = UNet(); net.load_state_dict(torch.load(ckpt, map_location=dev))
net.to(dev).eval(); logger.info("\u2713 loaded %s", ckpt)

# -------- sample --------------------------------------------------
ds  = XRayBeadDataset("data/train_list.txt", "data/raw/cam1.yaml", "data/raw/cam2.yaml")
smp = ds[row]

img1 = smp["image1"].unsqueeze(0).to(dev)
img2 = smp["image2"].unsqueeze(0).to(dev)
kp1  = as_list(smp["kp1"])
kp2  = as_list(smp["kp2"])
P1, P2 = smp["P1"], smp["P2"]

# -------- forward -------------------------------------------------
with torch.no_grad():
    h1, h2 = net(img1)[0, 0].cpu(), net(img2)[0, 0].cpu()
px1, py1 = map(int, peak_xy(h1));  px2, py2 = map(int, peak_xy(h2))

X, Y, Z  = triangulate(px1, py1, P1, px2, py2, P2)
rx1, ry1 = map(int, reproject((X, Y, Z), P1))
rx2, ry2 = map(int, reproject((X, Y, Z), P2))

# -------- save PNGs ----------------------------------------------
debug = Path("debug"); debug.mkdir(exist_ok=True)
cv2.imwrite(debug / "cam1_raw.png",  draw(smp["image1"][0], kp1, (px1, py1), (rx1, ry1)))
cv2.imwrite(debug / "cam2_raw.png",  draw(smp["image2"][0], kp2, (px2, py2), (rx2, ry2)))
cv2.imwrite(debug / "cam1_heat.png",
            cv2.applyColorMap((h1 * 255).byte().numpy(), cv2.COLORMAP_JET))
cv2.imwrite(debug / "cam2_heat.png",
            cv2.applyColorMap((h2 * 255).byte().numpy(), cv2.COLORMAP_JET))
logger.info("\u2713 wrote debug/cam1_raw.png cam2_raw.png cam1_heat.png cam2_heat.png")
