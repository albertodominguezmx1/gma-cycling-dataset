"""
Reuse example (Scientific Data).
Topographic comparison: crash-record locations vs. the segmented cycling network.

This script demonstrates how to reuse the dataset with a minimal, reproducible,
citable analysis:

  Question: do the 774 cyclist-involved crash records occur under topographic
  conditions different from those of the 13,380 cycling-network microsegments?

  Variables: elevation, mean slope, TRI, and roughness (GLO-30).
  Method:    medians, bootstrap 95% CI for median differences,
             two-sided Mann-Whitney U tests, and rank-biserial effect sizes.

Run in the same directory as the v2.2 pipeline outputs:
  python ejemplo_reuso.py

Inputs:
  - segmentos_ciclovias_enriched.csv
  - cyclist_accidents_enriched.csv

Outputs:
  - figura4_ejemplo_reuso.png
  - figura4_ejemplo_reuso.pdf
  - ejemplo_reuso_estadisticas.csv
  - ejemplo_reuso_resultados.txt
  - ejemplo_reuso_latex.txt        (ready-to-paste manuscript paragraph)
"""

from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
SEG_CSV = BASE / "segmentos_ciclovias_enriched.csv"
ACC_CSV = BASE / "cyclist_accidents_enriched.csv"

OUT_PNG = BASE / "figura4_ejemplo_reuso.png"
OUT_PDF = BASE / "figura4_ejemplo_reuso.pdf"
OUT_CSV = BASE / "ejemplo_reuso_estadisticas.csv"
OUT_TXT = BASE / "ejemplo_reuso_resultados.txt"
OUT_LATEX = BASE / "ejemplo_reuso_latex.txt"

N_BOOT = 5000
SEED = 42

VARIABLES = {
    "elevation_glo30": {
        "label": "Elevation",
        "unit": "m",
        "delta_unit": "m",
    },
    "slope_mean_glo30": {
        "label": "Mean slope",
        "unit": "% rise",
        "delta_unit": "percentage points",
    },
    "tri_glo30": {
        "label": "Terrain Ruggedness Index",
        "unit": "m",
        "delta_unit": "m",
    },
    "roughness_glo30": {
        "label": "Roughness",
        "unit": "m",
        "delta_unit": "m",
    },
}

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def fail(msg):
    log(f"ERROR: {msg}")
    OUT_TXT.write_text("\n".join(LOG), encoding="utf-8")
    sys.exit(1)


def ecdf(vals):
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def fmt_p_latex(p):
    if p < 0.001:
        return "$p < 0.001$"
    return f"$p = {p:.3f}$"


def main():
    log("=" * 78)
    log("REUSE EXAMPLE - CRASH RECORDS VS CYCLING NETWORK")
    log(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 78)

    for path in [SEG_CSV, ACC_CSV]:
        if not path.exists():
            fail(f"{path.name} was not found. Run the v2.2 pipeline first.")

    seg = pd.read_csv(SEG_CSV)
    acc = pd.read_csv(ACC_CSV)
    log(f"Segments: {len(seg):,}")
    log(f"Crash records: {len(acc):,}")
    log("Note: the crash-record unit is the person involved;")
    log("      this example is a methodological demonstration, not a causal analysis.")

    rng = np.random.default_rng(SEED)
    rows = []

    for var, meta in VARIABLES.items():
        if var not in seg.columns or var not in acc.columns:
            fail(f"Variable {var} is missing from one or both CSV files.")

        s = pd.to_numeric(seg[var], errors="coerce").dropna().to_numpy(dtype=float)
        a = pd.to_numeric(acc[var], errors="coerce").dropna().to_numpy(dtype=float)
        if len(s) == 0 or len(a) == 0:
            fail(f"Variable {var} has no valid values.")

        med_s = float(np.median(s))
        med_a = float(np.median(a))
        delta = med_a - med_s

        # Bootstrap 95% CI for the median difference (crash records - segments).
        boots = np.empty(N_BOOT, dtype=float)
        for i in range(N_BOOT):
            sb = rng.choice(s, size=len(s), replace=True)
            ab = rng.choice(a, size=len(a), replace=True)
            boots[i] = np.median(ab) - np.median(sb)
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

        mw = stats.mannwhitneyu(a, s, alternative="two-sided")
        u = float(mw.statistic)
        p = float(mw.pvalue)
        rank_biserial = 2.0 * u / (len(a) * len(s)) - 1.0

        row = {
            "variable": var,
            "label": meta["label"],
            "unit": meta["unit"],
            "n_segments": len(s),
            "n_crashes": len(a),
            "median_segments": med_s,
            "median_crashes": med_a,
            "delta_median_crash_minus_segment": delta,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "mannwhitney_u": u,
            "p_value": p,
            "rank_biserial": rank_biserial,
        }
        rows.append(row)

        log("")
        log(f"{meta['label']} ({var})")
        log(f"  Segments:  n={len(s):,}, median={med_s:.2f} {meta['unit']}, "
            f"mean={s.mean():.2f}")
        log(f"  Crash records: n={len(a):,}, median={med_a:.2f} {meta['unit']}, "
            f"mean={a.mean():.2f}")
        log(f"  Median difference (crashes - segments) = {delta:+.2f} {meta['delta_unit']} "
            f"[bootstrap 95% CI {ci_low:+.2f}, {ci_high:+.2f}]")
        log(f"  Mann-Whitney U = {u:,.0f}, {fmt_p(p)}, rank-biserial = {rank_biserial:+.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    log("")
    log(f"Statistics CSV: {OUT_CSV.name}")

    # ---- Figure 4: 2x2 ECDFs ----
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (_, row) in zip(axes, res.iterrows()):
        var = row["variable"]
        meta = VARIABLES[var]
        s = pd.to_numeric(seg[var], errors="coerce").dropna().to_numpy(dtype=float)
        a = pd.to_numeric(acc[var], errors="coerce").dropna().to_numpy(dtype=float)

        xs, ys = ecdf(s)
        xa, ya = ecdf(a)
        ax.step(xs, ys, where="post", color="#0077B6", lw=1.6,
                label=f"Network segments (n={len(s):,})")
        ax.step(xa, ya, where="post", color="#D62828", lw=1.6,
                label=f"Crash records (n={len(a):,})")

        ax.axvline(row["median_segments"], color="#0077B6", ls="--", lw=0.9, alpha=0.75)
        ax.axvline(row["median_crashes"], color="#D62828", ls="--", lw=0.9, alpha=0.75)

        # Robust x range to prevent extreme values from compressing the ECDF.
        lo, hi = np.percentile(np.concatenate([s, a]), [0.5, 99.5])
        if hi > lo:
            ax.set_xlim(lo, hi)

        ax.set_xlabel(f"{meta['label']} ({meta['unit']})", fontsize=9)
        ax.set_ylabel("Empirical CDF", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(color="0.90", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

        txt = (f"$\\Delta$median = {row['delta_median_crash_minus_segment']:+.2f} "
               f"{meta['delta_unit']}\n"
               f"95% CI [{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]\n"
               f"{fmt_p(row['p_value'])}, $r_{{rb}}$ = {row['rank_biserial']:+.3f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5,
                bbox=dict(facecolor="white", alpha=0.86, edgecolor="0.75", pad=2.2))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    log(f"Figure: {OUT_PNG.name} (300 dpi)")
    log(f"Figure: {OUT_PDF.name} (vector)")

    # ---- Ready-to-paste LaTeX paragraph for the manuscript ----
    r_slope = res.loc[res["variable"] == "slope_mean_glo30"].iloc[0]
    r_elev = res.loc[res["variable"] == "elevation_glo30"].iloc[0]
    r_tri = res.loc[res["variable"] == "tri_glo30"].iloc[0]
    r_rough = res.loc[res["variable"] == "roughness_glo30"].iloc[0]

    def dir_word(delta):
        return "higher" if delta > 0 else ("lower" if delta < 0 else "equal")

    paragraph = (
        "As a minimal reuse example, we compared the topographic conditions at the 774 "
        "crash records with those of the 13,380 network microsegments using the released "
        "CSV files alone (script \\texttt{ejemplo\\_reuso.py}; Figure~\\ref{fig:reuse}). "
        f"Crash locations showed {dir_word(r_slope['delta_median_crash_minus_segment'])} "
        f"median slope than network segments "
        f"({r_slope['median_crashes']:.2f}\\% vs.\\ {r_slope['median_segments']:.2f}\\% rise; "
        f"$\\Delta$ = {r_slope['delta_median_crash_minus_segment']:+.2f} percentage points, "
        f"bootstrap 95\\% CI [{r_slope['ci95_low']:+.2f}, {r_slope['ci95_high']:+.2f}]; "
        f"Mann--Whitney {fmt_p_latex(r_slope['p_value'])}, rank-biserial "
        f"$r$ = {r_slope['rank_biserial']:+.3f}). "
        f"Median elevation was {dir_word(r_elev['delta_median_crash_minus_segment'])} at crash "
        f"locations ({r_elev['median_crashes']:.1f} m vs.\\ {r_elev['median_segments']:.1f} m; "
        f"$\\Delta$ = {r_elev['delta_median_crash_minus_segment']:+.1f} m, 95\\% CI "
        f"[{r_elev['ci95_low']:+.1f}, {r_elev['ci95_high']:+.1f}]). "
        f"Corresponding median differences were "
        f"{r_tri['delta_median_crash_minus_segment']:+.2f} m for TRI and "
        f"{r_rough['delta_median_crash_minus_segment']:+.2f} m for roughness. "
        "This example is a methodological demonstration of reuse, not a causal risk analysis; "
        "the crash file is person-level, so multi-cyclist events contribute multiple records."
    )
    OUT_LATEX.write_text(paragraph + "\n", encoding="utf-8")
    log(f"LaTeX paragraph: {OUT_LATEX.name}")

    OUT_TXT.write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
