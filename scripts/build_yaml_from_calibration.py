#!/usr/bin/env python3
"""
Create cam1.yaml / cam2.yaml from the calibration numbers.
"""

import numpy as np, yaml, pathlib

OUT_DIR = pathlib.Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- your calibration numbers -----------------
K1 = np.array([[3462.5654603, 0,            546.625904256],
               [0,            3469.99045348,564.930836302],
               [0,            0,            1]], dtype=float)

R1 = np.array([[ 0.650439196848,-0.666746531664, 0.363837757411],
               [-0.450891793645, 0.0465457895967,0.891364167944],
               [-0.61124908313 , -0.743829652531,-0.270355333567]], dtype=float)

t1 = np.array([-2.71885159234,-7.17296534396,84.6864451925], dtype=float)

K2 = np.array([[3730.40410759, 0,            504.147588928],
               [0,            3731.73450418,498.052032482],
               [0,            0,            1]], dtype=float)

R2 = np.array([[-0.635426010546,-0.723656529644,-0.269360376123],
               [-0.391048432809, 0.000789324912, 0.920369762739],
               [-0.665818975637, 0.690159839508,-0.283486309389]], dtype=float)

t2 = np.array([16.1414230045,-5.73925159454,70.8357756669], dtype=float)
# -----------------------------------------------------

def make_P(K,R,t):
    return K @ np.hstack([R, t.reshape(3,1)])

def dump_yaml(P, path):
    with open(path, "w") as f:
        yaml.dump({"P": P.tolist()}, f)

dump_yaml(make_P(K1,R1,t1), OUT_DIR/"cam1.yaml")
dump_yaml(make_P(K2,R2,t2), OUT_DIR/"cam2.yaml")

print("✔  Wrote data/raw/cam1.yaml  and  data/raw/cam2.yaml")
