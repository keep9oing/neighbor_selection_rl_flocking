"""Cache median/IQR time profiles (phi, n_comp_r0, deg_mean, rank_dev) over
500-seed policy lanes, plus C1 degree profiles per L. Read-only over npz."""
import glob

import numpy as np

R3 = "/workspace/studies/acs-robust-r3-stress/data"
R2 = "/workspace/studies/acs-robust-r2/data"
OUT = "/workspace/figures/meeting/profiles.npz"
T = 1500  # profile window

LANES = {
    "C1_L250": f"{R3}/eval/C1_i80_L250_s500/*.npz",
    "R1_L250": f"{R2}/eval/R1_i110_L250_s500/*.npz",
    "C1_L125": f"{R3}/eval/C1_i80_L125_s500/*.npz",
    "C1_L500": f"{R3}/eval/C1_i80_L500_s500/*.npz",
}
KEYS = ["phi", "n_comp_r0", "deg_mean", "rank_dev"]

out = {}
for lane, pat in LANES.items():
    paths = sorted(glob.glob(pat))
    acc = {k: [] for k in KEYS}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for k in KEYS:
            if k in z.files:
                acc[k].append(z[k][:T + 1])
    for k in KEYS:
        if not acc[k]:
            continue
        a = np.stack(acc[k])  # (S, T+1)
        out[f"{lane}_{k}_med"] = np.nanmedian(a, axis=0)
        out[f"{lane}_{k}_q25"] = np.nanpercentile(a, 25, axis=0)
        out[f"{lane}_{k}_q75"] = np.nanpercentile(a, 75, axis=0)
    print(lane, len(paths), "files")

np.savez_compressed(OUT, **out)
print("saved", OUT)
