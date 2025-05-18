import numpy as np, torch


def _gauss_kernel(rad, sigma):
    a = np.arange(-rad, rad + 1)
    xx, yy = np.meshgrid(a, a)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2))


def generate_heatmap(keypoints, H, W, sigma=2):
    """
    keypoints : list[(x, y, bead_id)]  – bead_id ignored here
    returns   : torch.Tensor (1,1,H,W)
    """
    hm = np.zeros((H, W), np.float32)
    rad = int(3 * sigma);  g = _gauss_kernel(rad, sigma)

    for item in keypoints:
        if len(item) < 2:          # malformed tuple → skip
            continue
        x, y = item[0], item[1]
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - rad), min(W - 1, cx + rad)
        y0, y1 = max(0, cy - rad), min(H - 1, cy + rad)
        gx0, gx1 = x0 - (cx - rad), x1 - (cx - rad)
        gy0, gy1 = y0 - (cy - rad), y1 - (cy - rad)
        hm[y0:y1+1, x0:x1+1] = np.maximum(
            hm[y0:y1+1, x0:x1+1],
            g[gy0:gy1+1, gx0:gx1+1]
        )

    return torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


def softargmax_2d(hm: torch.Tensor):
    """Soft-argmax for a batch of heatmaps."""
    B, C, H, W = hm.shape
    probs = torch.softmax(hm.view(B, C, -1), dim=-1).view(B, C, H, W)
    xs = torch.linspace(0, W - 1, W, device=hm.device)
    ys = torch.linspace(0, H - 1, H, device=hm.device)
    xs = xs.view(1, 1, 1, W)
    ys = ys.view(1, 1, H, 1)
    x = (probs * xs).sum(dim=(2, 3))
    y = (probs * ys).sum(dim=(2, 3))
    return x.squeeze(1), y.squeeze(1)
