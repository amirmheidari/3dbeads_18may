#!/usr/bin/env python3
"""
Build data/raw/, data/labels/, data/train_list.txt from:
  * circ1.csv  (columns ..._cam1_X, ..._cam1_Y, ..._cam2_X, ..._cam2_Y)
  * Subject…_16709/*.jpg  (camera-1)
  * Subject…_16710/*.jpg  (camera-2)
"""

import csv, re, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# -------- USER paths --------------------------------------------------------
CSV_PATH    = Path("circ1.csv")      # or circ2.csv
CAM1_DIR    = Path("/Users/aheidari/Desktop/james_ai_training_15_may/subjec7R_no_motec_for_new_ai/CIRC/Subject7R_NoMotec_CIRC1_16709")
CAM2_DIR    = Path("/Users/aheidari/Desktop/james_ai_training_15_may/subjec7R_no_motec_for_new_ai/CIRC/Subject7R_NoMotec_CIRC1_16710")
OUT_RAW     = Path("data/raw")
OUT_LBL     = Path("data/labels")
LIST_OUT    = Path("data/train_list.txt")
COPY_IMAGES = True   # False = leave images in place and just reference them
# ---------------------------------------------------------------------------

OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

# 0) copy / link images ------------------------------------------------------
def copy_all(src_dir: Path):
    for jpg in src_dir.glob("*.jpg"):
        tgt = OUT_RAW / jpg.name
        if not tgt.exists():
            shutil.copy2(jpg, tgt)

if COPY_IMAGES:
    copy_all(CAM1_DIR)
    copy_all(CAM2_DIR)

# 1) load CSV ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# auto-detect bead names
beads = sorted(
    m.group(1)
    for col in df.columns
    if (m := re.match(r"(.+)_cam1_X$", col))
)
print("Beads found:", beads)

with open(LIST_OUT, "w") as list_f:
    for frame_idx, row in df.iterrows():
        fstr = f"{frame_idx:04d}"
        img1 = OUT_RAW / f"Subject7R_NoMotec_CIRC1_T3_16709_{fstr}.jpg"
        img2 = OUT_RAW / f"Subject7R_NoMotec_CIRC1_T3_16710_{fstr}.jpg"
        if not (img1.exists() and img2.exists()):
            continue  # skip if either jpg missing

        kp1, kp2 = [], []
        for bead in beads:
            x1, y1 = row[f"{bead}_cam1_X"], row[f"{bead}_cam1_Y"]
            x2, y2 = row[f"{bead}_cam2_X"], row[f"{bead}_cam2_Y"]
            if np.isfinite(x1) and np.isfinite(y1):
                kp1.append((x1, y1, bead))
            if np.isfinite(x2) and np.isfinite(y2):
                kp2.append((x2, y2, bead))

        txt1 = OUT_LBL / f"frame{fstr}_cam1.txt"
        txt2 = OUT_LBL / f"frame{fstr}_cam2.txt"
        with open(txt1, "w") as f:
            for x, y, bead in kp1:
                f.write(f"{x:.3f},{y:.3f},{bead}\n")
        with open(txt2, "w") as f:
            for x, y, bead in kp2:
                f.write(f"{x:.3f},{y:.3f},{bead}\n")

        list_f.write(f"raw/{img1.name},raw/{img2.name},labels/{txt1.name},labels/{txt2.name}\n")

print("✓  train_list.txt and label TXT files written in data/")
