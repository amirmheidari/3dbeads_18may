import numpy as np
import torch
from scripts.geometry import triangulate, triangulate_torch, reproject


def test_triangulate_torch_matches_numpy():
    P1 = np.eye(3, 4)
    P2 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    rng = np.random.default_rng(0)
    for _ in range(5):
        xyz = rng.normal(size=3)
        u1, v1 = reproject(xyz, P1)
        u2, v2 = reproject(xyz, P2)
        np_xyz = np.asarray(triangulate(u1, v1, P1, u2, v2, P2))
        t_xyz = triangulate_torch(torch.tensor(u1), torch.tensor(v1), P1,
                                  torch.tensor(u2), torch.tensor(v2), P2)
        assert np.allclose(np_xyz, t_xyz.numpy(), atol=1e-3)

