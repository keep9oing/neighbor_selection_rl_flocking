"""Pre-registered analysis for acs-robust-r3-stress (see PROBLEM.md).

Sections
  0  Sanity: reproduce r2's published k12 n=500 success counts (460/455).
  1  Frontier table (n=500): fails + Wilson CI, succ-J med, CVaR10, q90.
  2  H-A verdict: policy x L x arm(k13/14/16/18) per PROBLEM operational rule.
  3  H-B verdict: b1 k7@N10, b2 k26@N40, b3 rule-k reliability + J-grid slice.
  4  H-C verdict: C1 in-pool insurance (pooled <=1% + per-L McNemar vs k12).
  5  Extrapolation refs (32s): k12/k16/k19 @ L375/L750 vs policy evals.
  6  Failure-mode census: final component split for every failed episode.
  7  audit_one detail rows for all paired comparisons -> stress_stats.csv.

Conventions identical to r2 audit_stats.py (audit_one imported from it).
Missing inputs (e.g. FC n=500 lanes still running) -> WARN + skip, so the
script can run incrementally; the final report run must show no WARNs.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

R3 = "/workspace/studies/acs-robust-r3-stress"
R2 = "/workspace/studies/acs-robust-r2"
OLD = "/workspace/studies/acs-robust-train/data/eval"
FRONT = "/workspace/studies/acs-c2-train/data/frontier_L.csv"
sys.path.insert(0, f"{R2}/src")
from audit_stats import audit_one  # noqa: E402  (same paired-stat conventions)

WARNS = []


def warn(msg):
    WARNS.append(msg)
    print(f"WARN: {msg}")


def load_csv(path, what):
    if not os.path.exists(path):
        warn(f"missing {what}: {path}")
        return None
    return pd.read_csv(path)[["seed", "J", "success"]]


def ref(k, L, N=20, n500=False):
    """k-NN reference; r3 first, r2 fallback (k12@L250/500 n=500 live in r2)."""
    for base in (f"{R3}/data/knnref", f"{R2}/data/knnref"):
        p = f"{base}/k{k}_L{L:g}_N{N}_summary.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            if n500 and len(d) < 500:
                continue  # gate remnant (e.g. r3 k12@L250 8-seed) — keep looking
            return d[["seed", "J", "success"]]
    warn(f"missing knnref k{k} L{L:g} N{N}")
    return None


def pol(label, study=R3):
    return load_csv(f"{study}/data/eval/{label}_summary.csv", f"policy {label}")


def wilson(f, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = f / n
    den = 1 + z * z / n
    ctr = ph + z * z / (2 * n)
    adj = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((ctr - adj) / den, (ctr + adj) / den)


def cvar10(j_succ):
    j = np.sort(np.asarray(j_succ))
    m = max(1, int(math.ceil(0.1 * len(j))))
    return float(np.mean(j[-m:]))


def frontier_row(name, df):
    n = len(df)
    f = int((df.success == 0).sum())
    js = df[df.success == 1].J.values
    lo, hi = wilson(f, n)
    return dict(arm=name, n=n, fails=f, fail_pct=100 * f / n,
                wilson_lo=100 * lo, wilson_hi=100 * hi,
                J_med=float(np.median(js)), J_q90=float(np.percentile(js, 90)),
                CVaR10=cvar10(js))


def mcnemar(poldf, refdf):
    p = poldf.set_index("seed")
    r = refdf.set_index("seed")
    c = p.index.intersection(r.index)
    p, r = p.loc[c], r.loc[c]
    b = int(((p.success == 0) & (r.success == 1)).sum())
    cc = int(((p.success == 1) & (r.success == 0)).sum())
    return (b, cc, stats.binomtest(b, b + cc, 0.5).pvalue if b + cc else np.nan)


def paired_wilcoxon(poldf, refdf):
    """(median dJ, wilcoxon p, n) on co-success pairs; dJ = pol - ref."""
    p = poldf.set_index("seed")
    r = refdf.set_index("seed")
    c = p.index.intersection(r.index)
    p, r = p.loc[c], r.loc[c]
    both = (p.success == 1) & (r.success == 1)
    d = (p.J[both] - r.J[both]).values
    if len(d) < 6:
        return (np.nan, np.nan, len(d))
    return (float(np.median(d)), float(stats.wilcoxon(d).pvalue), len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default=f"{R3}/data/stress_stats.csv")
    ap.add_argument("--census", action="store_true",
                    help="run failure-mode census (reads every failed npz)")
    args = ap.parse_args()
    rows = []

    print("=== 0. sanity: r2 k12 n=500 reproduction ===")
    for L, want in ((250, 460), (500, 455)):
        d = ref(12, L, n500=True)
        got = int(d.success.sum())
        tag = "OK" if got == want else "MISMATCH"
        print(f"k12 L{L}: success {got}/500 (published {want}) {tag}")
        assert got == want, f"sanity failed for k12 L{L}"

    print("\n=== 1. frontier table (n=500 unless noted) ===")
    fr = []
    arms = {}
    for L in (125, 250, 500):
        for k in (12, 13, 14, 16, 18, 19):
            d = ref(k, L, n500=True)
            if d is None or len(d) < 500:
                if k == 19:
                    warn(f"FC(k19) L{L} n=500 not ready")
                continue
            arms[(k, L)] = d
            fr.append(frontier_row(f"k{k}_L{L}", d))
        for lab, name in ((f"C1_i80_L{L}_s500", f"C1_L{L}"),
                          (f"R1_i110_L{L}_s500", f"R1_L{L}"),
                          (f"A60_L{L}_s500", f"A60_L{L}")):
            for study in (R3, R2):
                if os.path.exists(f"{study}/data/eval/{lab}_summary.csv"):
                    d = pol(lab, study)
                    arms[(name, L)] = d
                    fr.append(frontier_row(name, d))
                    break
    frdf = pd.DataFrame(fr)
    pd.set_option("display.width", 200)
    print(frdf.to_string(index=False,
                         float_format=lambda x: f"{x:8.1f}"))

    print("\n=== 2. H-A verdict (pre-registered) ===")
    # (i) reliability: fail_A <= 5/500 at BOTH L250 & L500
    rel_ok = {}
    for k in (13, 14, 16, 18):
        fails = {L: int((arms[(k, L)].success == 0).sum()) for L in (250, 500)
                 if (k, L) in arms}
        ok = len(fails) == 2 and all(v <= 5 for v in fails.values())
        rel_ok[k] = ok
        print(f"  arm k{k}: fails L250={fails.get(250, '?')} L500={fails.get(500, '?')} -> criterion(i) {'MET' if ok else 'not met'}")
    ha = {}
    for pname, plabels in (("R1", {250: ("R1_i110_L250_s500", R2), 500: ("R1_i110_L500_s500", R2)}),
                           ("C1", {250: ("C1_i80_L250_s500", R3), 500: ("C1_i80_L500_s500", R3)})):
        for L in (250, 500):
            lab, study = plabels[L]
            pdf = pol(lab, study)
            refuted = False
            for k in (13, 14, 16, 18):
                if (k, L) not in arms:
                    continue
                med, wp, n = paired_wilcoxon(pdf, arms[(k, L)])
                audit_one(f"{pname} vs k{k} L{L} n500", pdf, arms[(k, L)], rows)
                if rel_ok[k] and med > 0 and wp < 0.05:
                    refuted = True
                    print(f"  -> H-A REFUTED for {pname}@L{L} by k{k}")
            k12d = arms.get((12, L))
            if k12d is not None:
                audit_one(f"{pname} vs k12 L{L} n500", pdf, k12d, rows)
            ha[(pname, L)] = refuted
            print(f"  H-A for {pname}@L{L}: {'REFUTED' if refuted else 'SURVIVES'}")
    full_refute = all(ha.values())
    print(f"H-A OVERALL: {'FULLY REFUTED' if full_refute else 'SURVIVES (no arm meets (i)+(ii) everywhere)'}")

    print("\n=== 3. H-B verdict (ratio rule) ===")
    c1n10 = pol("C1_i80_N10L177_s32", R2)
    r1n10 = pol("R1_i110_N10L177_s32", R2)
    k7 = ref(7, 177, 10)
    b1 = False
    if c1n10 is not None and k7 is not None:
        med, wp, n = paired_wilcoxon(c1n10, k7)   # dJ = C1 - k7
        b, cc, mp = mcnemar(c1n10, k7)
        audit_one("C1 vs k7(.65N) N10", c1n10, k7, rows)
        if r1n10 is not None:
            audit_one("R1 vs k7(.65N) N10", r1n10, k7, rows)
        b1 = (med < 0 and wp < 0.05) or (mp < 0.05 and b < cc)
        print(f"  b1 (C1 beats k7@N10): med dJ {med:+.1f} w_p {wp:.3g}, McNemar b={b} cc={cc} p={mp:.3g} -> {'MET' if b1 else 'NOT MET'}")
    k26 = ref(26, 354, 40)
    if k26 is not None:
        for lab, nm in (("C1_i80_N40L354_s32", "C1"), ("R1_i110_N40L354_s32", "R1")):
            d = pol(lab, R2)
            if d is not None:
                audit_one(f"{nm} vs k26(.65N) N40", d, k26, rows)
    rule_unrel = False
    print("  b3 rule-k reliability + 32-seed J-grid slice (seeds 1000-1031):")
    for k in (13, 14):
        for L in (125, 250, 500):
            if (k, L) not in arms:
                continue
            d = arms[(k, L)]
            f = int((d.success == 0).sum())
            if f > 5:
                rule_unrel = True
            s32 = d[d.seed <= 1031]
            jm = s32[s32.success == 1].J.median()
            print(f"    k{k} L{L}: fails {f}/500 ({100 * f / 500:.1f}%); 32-slice succ {int(s32.success.sum())}/32 J_med {jm:.1f}")
    hb = b1 and rule_unrel
    print(f"H-B: b1={'MET' if b1 else 'NOT MET'}, rule-k unreliable somewhere={'YES' if rule_unrel else 'NO'} -> {'SURVIVES' if hb else 'WEAKENED/REFUTED'}")

    print("\n=== 4. H-C verdict (C1 in-pool insurance) ===")
    tot_f, tot_n, percl = 0, 0, []
    for L in (125, 250, 500):
        c1 = arms.get((f"C1_L{L}", L))
        k12 = arms.get((12, L))
        if c1 is None or k12 is None:
            warn(f"H-C leg L{L} incomplete")
            continue
        f = int((c1.success == 0).sum())
        tot_f, tot_n = tot_f + f, tot_n + len(c1)
        b, cc, mp = mcnemar(c1, k12)
        sig = (mp < 0.05) and (b < cc)
        percl.append(sig)
        print(f"  L{L}: C1 fails {f}/500; McNemar vs k12 b={b} cc={cc} p={mp:.3g} {'SIG' if sig else 'n.s.'}")
    hc = (tot_n > 0) and (tot_f / tot_n <= 0.01) and all(percl) and len(percl) == 3
    print(f"H-C: pooled {tot_f}/{tot_n} ({100 * tot_f / max(tot_n, 1):.2f}%) -> {'MET' if hc else 'NOT MET'}")

    print("\n=== 5. extrapolation refs (32 seeds) ===")
    for L in (375, 750):
        for k in (12, 16, 19):
            d = ref(k, L)
            if d is None:
                continue
            js = d[d.success == 1].J
            print(f"  k{k} L{L}: succ {int(d.success.sum())}/32 J_med {js.median():.1f}")
        for lab, study, nm in ((f"C1_i80_L{L}_s32", R2, "C1"),
                               (f"R1_i110_L{L}_s32", OLD.rsplit('/data', 1)[0], "R1"),
                               (f"A60_L{L}_s32", OLD.rsplit('/data', 1)[0], "A60")):
            d = pol(lab, study)
            if d is not None:
                js = d[d.success == 1].J
                print(f"  {nm} L{L}: succ {int(d.success.sum())}/32 J_med {js.median():.1f}")
                k12r = ref(12, L)
                if k12r is not None:
                    audit_one(f"{nm} vs k12 L{L} extrap", d, k12r, rows)

    if args.census:
        print("\n=== 6. failure-mode census (final component split) ===")
        census(arms)

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    print(f"\n-> {args.csv_out}")
    print(f"WARNS: {len(WARNS)}")
    for w in WARNS:
        print(f"  - {w}")


def comp_sizes(pos, r0):
    n = len(pos)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    adj = csr_matrix(d2 < r0 * r0)
    nc, lab = connected_components(adj, directed=False)
    return sorted(np.bincount(lab).tolist(), reverse=True)


def census(arms):
    pats = [f"{R3}/data/knnref", f"{R2}/data/knnref",
            f"{R3}/data/eval", f"{R2}/data/eval"]
    for (key, L), df in sorted(arms.items(), key=lambda x: str(x[0])):
        fails = df[df.success == 0].seed.tolist()
        if not fails:
            continue
        splits = {}
        for s in fails:
            cond = f"k{key}_L{L:g}_N20" if isinstance(key, int) else None
            if cond:
                cands = [f"{b}/{cond}/{cond}_s{s}.npz" for b in pats[:2]]
            else:
                lab = {"C1": f"C1_i80_L{L}_s500", "R1": f"R1_i110_L{L}_s500",
                       "A60": f"A60_L{L}_s500"}[key.split("_")[0]]
                cands = [f"{b}/{lab}/{lab}_s{s}.npz" for b in pats[2:]]
            p = next((c for c in cands if os.path.exists(c)), None)
            if p is None:
                warn(f"census: npz missing for {key} L{L} s{s}")
                continue
            z = np.load(p, allow_pickle=True)
            m = json.loads(str(z["meta"]))
            sizes = comp_sizes(z["pos_snaps"][-1], m.get("r0", 60))
            splits[s] = sizes
        hist = {}
        for s, sz in splits.items():
            kk = "+".join(map(str, sz))
            hist[kk] = hist.get(kk, 0) + 1
        print(f"  {key} L{L}: {len(splits)} fails -> " +
              ", ".join(f"[{k}]x{v}" for k, v in
                        sorted(hist.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
