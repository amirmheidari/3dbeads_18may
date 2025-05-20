import torch
from scripts.heatmaps_multi import generate_multichannel_heatmaps, IDS


def test_generate_multichannel_heatmaps_peak():
    H = W = 16
    for i, bead in enumerate(IDS):
        hm = generate_multichannel_heatmaps([(5, 7, bead)], H, W, sigma=1)
        y, x = divmod(int(hm[i].argmax()), W)
        assert (x, y) == (5, 7)

