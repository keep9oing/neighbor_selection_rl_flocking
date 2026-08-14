"""Paper-table emitter (study acs-confirm): markdown tables from summary CSVs.

Table 1 — fresh-seed (1500-1999) N=20 confirmation grid.
Table 2 — N-axis (1000-1499) completion grid at n=500.
Reuses confirm_judge loaders/stats so every number shares one code path.
"""
import numpy as np

from confirm_judge import (FRESH_N20, NAXIS, R3, cvar10, knn, mcnemar, pair,
                           pol, wilson)


def row(name, df, ref=None):
    n = len(df)
    f = int((df.success == 0).sum())
    lo, hi = wilson(f, n)
    jm = df.J[df.success == 1].median()
    cv = cvar10(df.J[df.success == 1])
    mc = ""
    if ref is not None:
        a, b = pair(df, ref)
        _, _, p = mcnemar(a, b)
        mc = f"{p:.2e}" if p < 0.01 else f"{p:.3f}"
    return (f"| {name} | {f}/{n} ({100 * f / n:.1f}%) | "
            f"[{100 * lo:.1f}, {100 * hi:.1f}]% | {mc} | {jm:.1f} | {cv:.1f} |")


def main():
    print("### 표 1 — 신선-시드 확증 그리드 (N=20, 시드 1500–1999, n=500/암)\n")
    print("| 암 | 실패 | Wilson 95% CI | McNemar vs k12 | 성공-J 중앙값 | CVaR10 |")
    print("|---|---|---|---|---|---|")
    for L in (125, 250, 500):
        k12 = knn(12, L, 20, FRESH_N20)
        arms = [(f"π_E @L{L}", pol(f"piE_L{L}", FRESH_N20), k12),
                (f"π_R @L{L}", pol(f"piR_L{L}", FRESH_N20), k12)]
        for k in {125: (11, 12, 13), 250: (12, 13), 500: (10, 12, 13)}[L]:
            arms.append((f"k{k} @L{L}", knn(k, L, 20, FRESH_N20),
                         k12 if k != 12 else None))
        arms.append((f"FC(k19) @L{L}", knn(19, L, 20, FRESH_N20), k12))
        for name, df, ref in arms:
            print(row(name, df, ref))

    print("\n### 표 2 — N축 완성 그리드 (시드 1000–1499, n=500/암; 참조는 r3 아카이브)\n")
    print("| 암 | 실패 | Wilson 95% CI | McNemar vs 최선-k | 성공-J 중앙값 | CVaR10 |")
    print("|---|---|---|---|---|---|")
    n10ref = knn(6, 177, 10, NAXIS, R3)
    n40ref = knn(24, 354, 40, NAXIS, R3)
    print(row("π_E @N10,L177", pol("piE_N10L177", NAXIS), n10ref))
    print(row("π_R @N10,L177", pol("piR_N10L177", NAXIS), n10ref))
    print(row("k6 @N10 (r3)", n10ref))
    print(row("k7 @N10 (r3)", knn(7, 177, 10, NAXIS, R3)))
    print(row("k8 @N10 (r3)", knn(8, 177, 10, NAXIS, R3)))
    print(row("FC(k9) @N10 (신규)", knn(9, 177, 10, NAXIS)))
    print(row("π_E @N40,L354", pol("piE_N40L354", NAXIS), n40ref))
    print(row("π_R @N40,L354", pol("piR_N40L354", NAXIS), n40ref))
    print(row("k24 @N40 (r3)", n40ref))
    print(row("k26 @N40 (r3)", knn(26, 354, 40, NAXIS, R3)))
    print(row("k28 @N40 (r3)", knn(28, 354, 40, NAXIS, R3)))
    print(row("FC(k39) @N40 (신규)", knn(39, 354, 40, NAXIS)))

    print("\n### ablation (π_E nearest-projection, 신선 시드)\n")
    print("| 암 | 실패 | Wilson 95% CI | McNemar vs 원본 π_E | 성공-J 중앙값 | CVaR10 |")
    print("|---|---|---|---|---|---|")
    for L in (250, 125):
        print(row(f"proj(π_E) @L{L}", pol(f"ablE_L{L}", FRESH_N20),
                  pol(f"piE_L{L}", FRESH_N20)))
        print(row(f"π_E @L{L} (원본)", pol(f"piE_L{L}", FRESH_N20)))


if __name__ == "__main__":
    main()
