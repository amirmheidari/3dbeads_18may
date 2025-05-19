#!/usr/bin/env python3
import cv2, yaml, torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from scripts.logging_utils import setup_logger

logger = setup_logger(__name__)


class XRayBeadDataset(Dataset):
    """
    list_file rows:
        img_cam1.png,img_cam2.png,kp_cam1.txt,kp_cam2.txt
    key-point TXT rows:
        x, y, bead_id
    """

    def __init__(self, list_file, cam1_yaml, cam2_yaml,
                 root="data", transform=None):
        self.root      = Path(root)
        with open(list_file, "r", encoding="utf-8") as f:
            self.pairs = [ln.strip().split(",") for ln in f]
        self.P1        = self._load_P(cam1_yaml)   # (3,4)
        self.P2        = self._load_P(cam2_yaml)
        self.transform = transform

        logger.info(
            "Loaded %d image pairs from %s", len(self.pairs), list_file
        )

    @staticmethod
    def _load_P(path):
        with open(path, "r", encoding="utf-8") as f:
            P = np.asarray(yaml.safe_load(f)["P"], np.float32)
        if P.shape != (3, 4):
            raise ValueError(f"{path}: projection must be 3×4, got {P.shape}")
        return P

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_fp, img2_fp, kp1_fp, kp2_fp = self.pairs[idx]
        logger.debug("Loading sample %d: %s %s", idx, img1_fp, img2_fp)

        # ----- images --------------------------------------------------
        img1 = cv2.imread(str(self.root / img1_fp), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(self.root / img2_fp), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            logger.error("Missing %s or %s", img1_fp, img2_fp)
            raise FileNotFoundError(f"Missing {img1_fp} or {img2_fp}")

        img1 = torch.from_numpy(img1.astype("float32") / 255).unsqueeze(0)
        img2 = torch.from_numpy(img2.astype("float32") / 255).unsqueeze(0)
        if self.transform:
            img1, img2 = self.transform(img1, img2)

        # ----- key-points ---------------------------------------------
        def read_txt(path):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) < 2:
                        continue
                    try:
                        x, y = float(parts[0]), float(parts[1])
                    except ValueError:
                        continue
                    bead = parts[2] if len(parts) >= 3 else ""
                    rows.append((x, y, bead))
            return rows                    # always list[tuple]

        kp1 = read_txt(self.root / kp1_fp)
        if len(kp1) == 0:
            logger.warning("No keypoints in %s", kp1_fp)
        kp2 = read_txt(self.root / kp2_fp)
        if len(kp2) == 0:
            logger.warning("No keypoints in %s", kp2_fp)
        logger.debug(
            "Sample %d keypoints: cam1=%d cam2=%d", idx, len(kp1), len(kp2)
        )

        return {
            "image1": img1,
            "image2": img2,
            "kp1":    kp1,
            "kp2":    kp2,
            "P1":     self.P1,
            "P2":     self.P2,
        }
