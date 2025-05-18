#!/usr/bin/env python3
import numpy as np
import torch


def _reshape_P(P):
    """Accept (3,4), (1,3,4) or flat 12-vector → (3,4) or None."""
    P = np.asarray(P, np.float64).reshape(-1)
    if P.size == 12:
        return P.reshape(3, 4)
    return None


def triangulate(u1, v1, P1, u2, v2, P2):
    P1, P2 = _reshape_P(P1), _reshape_P(P2)
    if P1 is None or P2 is None:
        return (np.nan, np.nan, np.nan)

    A = np.stack([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1],
    ])

    if not np.all(np.isfinite(A)):
        return (np.nan, np.nan, np.nan)

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return (np.nan, np.nan, np.nan)

    Xh = Vt[-1]
    if Xh.size != 4 or np.isclose(Xh[3], 0):
        return (np.nan, np.nan, np.nan)

    Xh /= Xh[3]
    return tuple(Xh[:3])


def reproject(XYZ, P):
    P = _reshape_P(P)
    if P is None:
        return (np.nan, np.nan)
    X, Y, Z = XYZ
    uvw = P @ np.array([X, Y, Z, 1.0])
    if np.isclose(uvw[2], 0):
        return (np.nan, np.nan)
    return (uvw[0] / uvw[2], uvw[1] / uvw[2])


def triangulate_torch(u1, v1, P1, u2, v2, P2, eps=1e-9):
    """Direct Linear Transform in PyTorch (differentiable)."""
    dev  = u1.device if torch.is_tensor(u1) else torch.device("cpu")
    dtype = u1.dtype if torch.is_tensor(u1) else torch.float32
    P1 = torch.as_tensor(P1, dtype=dtype, device=dev).reshape(3, 4)
    P2 = torch.as_tensor(P2, dtype=dtype, device=dev).reshape(3, 4)

    A = torch.stack([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1],
    ])

    U, S, Vh = torch.linalg.svd(A)
    Xh = Vh[-1]
    Xh = Xh / (Xh[3] + eps)
    return Xh[:3]


def reproject_torch(XYZ, P):
    """Project a 3D point using torch ops."""
    XYZ = torch.as_tensor(XYZ)
    dev  = XYZ.device
    dtype = XYZ.dtype
    P = torch.as_tensor(P, dtype=dtype, device=dev).reshape(3, 4)
    XYZ1 = torch.cat([XYZ, XYZ.new_ones(1)])
    uvw = P @ XYZ1
    return uvw[0] / uvw[2], uvw[1] / uvw[2]
