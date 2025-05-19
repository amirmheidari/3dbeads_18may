import torch
import math

IDS = ['B0','B1','B2','B3','B4','B5']   # order must stay fixed


def draw_gaussian(hm, x, y, sigma=3):
    H, W = hm.shape
    xs = torch.arange(W, device=hm.device)
    ys = torch.arange(H, device=hm.device).view(-1,1)
    heat = torch.exp(-((xs-x)**2 + (ys-y)**2)/(2*sigma**2))
    hm[:] = torch.maximum(hm, heat)


def generate_multichannel_heatmaps(kps, H, W, sigma=3):
    """
    kps : list[(x, y, label:str)]   len ≤ K
    returns tensor [K, H, W]
    """
    hm = torch.zeros(len(IDS), H, W)
    for x, y, label in kps:
        ch = IDS.index(label)
        draw_gaussian(hm[ch], x, y, sigma)
    return hm
