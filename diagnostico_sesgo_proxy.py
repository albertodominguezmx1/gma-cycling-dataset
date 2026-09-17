"""
Bias diagnostic for vertical_complexity_proxy - GMA dataset v2.2.

Purpose: quantify the systematic offset of the proxy (glo30 - nasadem),
determine whether it dominates any vertical-cover/structure signal, and evaluate
the effect of a trial bias correction across reference zones.

This script does not modify the data; it is read-only. Any final correction
would be integrated into the pipeline only after reviewing these results.

Inputs (same directory):
  - segmentos_ciclovias_enriched_v2.csv   (required)
  - cyclist_accidents_enriched_v2.csv     (optional)
Outputs:
  - diagnostico_sesgo_proxy.txt           (report)
  - fig_histograma_proxy.png              (before/after distribution)
  - fig_boxplot_zonas.png                 (proxy by zone, before/after)
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEGMENTS = 'segmentos_ciclovias_enriched_v2.csv'
CRASH_RECORDS = 'cyclist_accidents_enriched_v2.csv'
REPORT = 'diagnostico_sesgo_proxy.txt'

# Current paper thresholds (evaluated as-is on the corrected proxy)
TH_LOW, TH_HIGH = 2.0, 8.0

# Reference zones (same official coordinates as the v2.2 audit)
ZONES = {
    'Bosque_La_Primavera': (-103.7000, 20.5333, -103.4667, 20.7333),
    'Centro_Historico_GDL': (-103.3550, 20.6680, -103.3390, 20.6840),
    'Periferico_Norte': (-103.3580, 20.7250, -103.3460, 20.7370),
}

LINES = []

def log(msg=''):
    print(msg)
    LINES.append(msg)

def proxy_stats(s, label):
    log(f"  [{label}] n={len(s)}, mean={s.mean():.2f}, median={s.median():.2f}, "
        f"std={s.std():.2f}, p5={s.quantile(0.05):.2f}, p25={s.quantile(0.25):.2f}, "
        f"p75={s.quantile(0.75):.2f}, p95={s.quantile(0.95):.2f}")
    log(f"  [{label}] negative: {(s < 0).sum()} ({(s < 0).mean()*100:.1f}%), "
        f"positive: {(s > 0).sum()} ({(s > 0).mean()*100:.1f}%)")

def categorize(v):
    a = abs(v)
    if a < TH_LOW:
        return 'low'
    elif a < TH_HIGH:
        return 'moderate'
    return 'high'

def flag_distribution(series, label):
    vc = series.map(categorize).value_counts()
    total = len(series)
    parts = ', '.join(f"{k}={v} ({v/total*100:.1f}%)" for k, v in vc.items())
    log(f"  [{label}] {parts}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("BIAS DIAGNOSTIC FOR vertical_complexity_proxy")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    if not os.path.exists(SEGMENTS):
        print(f"ERROR: {SEGMENTS} was not found. Run this script in the pipeline directory.")
        sys.exit(1)

    seg = pd.read_csv(SEGMENTS)
    seg['lon_mid'] = (seg['Inicio_Lon'] + seg['Fin_Lon']) / 2
    seg['lat_mid'] = (seg['Inicio_Lat'] + seg['Fin_Lat']) / 2
    proxy = seg['vertical_complexity_proxy']
    log(f"\nSegments loaded: {len(seg)}")

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("A. GLOBAL PROXY DISTRIBUTION (SEGMENTS)")
    log("=" * 70)
    proxy_stats(proxy, 'global')

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("B. SYSTEMATIC OFFSET ESTIMATION (3 METHODS)")
    log("=" * 70)

    # B.1 "Flat and smooth" subset: probable bare urban terrain.
    # Percentile-based criteria (robust to each feature scale).
    s25 = seg['slope_mean_glo30'].quantile(0.25)
    r25 = seg['roughness_glo30'].quantile(0.25)
    t25 = seg['tri_glo30'].quantile(0.25)
    flat = seg[(seg['slope_mean_glo30'] <= s25) &
                (seg['roughness_glo30'] <= r25) &
                (seg['tri_glo30'] <= t25)]
    off_flat = flat['vertical_complexity_proxy'].median()
    log(f"\nB.1 Flat+smooth subset (slope<=p25={s25:.2f}, rough<=p25={r25:.2f}, tri<=p25={t25:.2f}):")
    log(f"    n={len(flat)} segments, proxy median = {off_flat:.2f} m  <- primary candidate")

    # B.2 Reference road corridor (Periferico Norte)
    b = ZONES['Periferico_Norte']
    perif = seg[(seg['lon_mid'] >= b[0]) & (seg['lon_mid'] <= b[2]) &
                (seg['lat_mid'] >= b[1]) & (seg['lat_mid'] <= b[3])]
    off_perif = perif['vertical_complexity_proxy'].median()
    log(f"\nB.2 Periferico Norte corridor (flat terrain, independent reference):")
    log(f"    n={len(perif)} segments, proxy median = {off_perif:.2f} m  <- control candidate")

    # B.3 Global median (biased by cover; reference only)
    off_global = proxy.median()
    log(f"\nB.3 Global median (includes cover; reference only):")
    log(f"    proxy median = {off_global:.2f} m")

    offset = off_flat
    log(f"\n>>> OFFSET ADOPTED for the trial correction: {offset:.2f} m (B.1)")
    log("    (if B.1 and B.2 differ substantially, report it: it would indicate a spatial gradient)")

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("C. SPATIAL STRUCTURE OF THE PROXY (SMOOTH OFFSET VS IRREGULAR SIGNAL)")
    log("=" * 70)
    # A datum/geoid offset varies smoothly in space; the vegetation/building
    # signal is irregular. Five-km blocks:
    seg['bx'] = (seg['lon_mid'] * 111320).floordiv(5000)
    seg['by'] = (seg['lat_mid'] * 110540).floordiv(5000)
    blocks = seg.groupby(['bx', 'by'])['vertical_complexity_proxy'].agg(['mean', 'count'])
    blocks = blocks[blocks['count'] >= 20]
    log(f"~5 km blocks with >=20 segments: {len(blocks)}")
    log(f"Block-mean range: [{blocks['mean'].min():.2f}, {blocks['mean'].max():.2f}] m")
    log(f"Between-block spread (SD of means): {blocks['mean'].std():.2f} m")
    log("Reading: if block means are all similar and negative,")
    log("the dominant term is a spatially smooth datum offset, not cover.")

    # Correlations: a constant offset correlates with nothing; the vertical
    # structure signal should correlate (even weakly) with roughness/TRI and
    # NOT with absolute elevation.
    log("\nPearson correlations of the proxy (segments):")
    for col in ['elevation_glo30', 'slope_mean_glo30', 'tri_glo30', 'roughness_glo30']:
        if col in seg.columns:
            log(f"  proxy vs {col}: r={seg['vertical_complexity_proxy'].corr(seg[col]):.3f}")

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("D. REFERENCE ZONES: BEFORE CORRECTION")
    log("=" * 70)
    zones_before = {}
    for name, bb in ZONES.items():
        sub = seg[(seg['lon_mid'] >= bb[0]) & (seg['lon_mid'] <= bb[2]) &
                  (seg['lat_mid'] >= bb[1]) & (seg['lat_mid'] <= bb[3])]
        zones_before[name] = sub['vertical_complexity_proxy']
        if len(sub) == 0:
            log(f"\n{name}: 0 segments (no network coverage in zone)")
            continue
        log(f"\n{name}: {len(sub)} segments")
        proxy_stats(sub['vertical_complexity_proxy'], name)

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log(f"E. TRIAL CORRECTION: proxy_corr = proxy - ({offset:.2f})")
    log("=" * 70)
    seg['proxy_corr'] = seg['vertical_complexity_proxy'] - offset
    proxy_stats(seg['proxy_corr'], 'global corrected')

    log("\nFlag distribution with current thresholds (|2|, |8| m):")
    flag_distribution(seg['vertical_complexity_proxy'], 'BEFORE')
    flag_distribution(seg['proxy_corr'], 'AFTER')

    log("\nReference zones: AFTER correction")
    zones_after = {}
    for name, bb in ZONES.items():
        sub = seg[(seg['lon_mid'] >= bb[0]) & (seg['lon_mid'] <= bb[2]) &
                  (seg['lat_mid'] >= bb[1]) & (seg['lat_mid'] <= bb[3])]
        zones_after[name] = sub['proxy_corr']
        if len(sub) == 0:
            continue
        log(f"\n{name}: {len(sub)} segments")
        proxy_stats(sub['proxy_corr'], name)
        flag_distribution(sub['proxy_corr'], f"{name} flags")

    # Key contrast: forest vs flat corridor
    if len(zones_after['Bosque_La_Primavera']) and len(zones_after['Periferico_Norte']):
        contrast = (zones_after['Bosque_La_Primavera'].median() -
                     zones_after['Periferico_Norte'].median())
        log(f"\n>>> KEY CONTRAST (corrected forest median - corridor): {contrast:.2f} m")
        log("    Reading: if the contrast remains near 0, the proxy does NOT discriminate")
        log("    forest from flat terrain at segment level and the variable must be")
        log("    redesigned; if clearly positive, the correction works.")
        log("    NOTE: the cycling network covers only the northeastern edge of the protected area,")
        log("    not the forest interior; interpret with that caveat.")

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("F. CRASH RECORDS (OPTIONAL)")
    log("=" * 70)
    acc_stats = None
    if os.path.exists(CRASH_RECORDS):
        acc = pd.read_csv(CRASH_RECORDS)
        proxy_stats(acc['vertical_complexity_proxy'], 'crash records global')
        acc_corr = acc['vertical_complexity_proxy'] - offset
        proxy_stats(acc_corr, 'crash records corrected')
        flag_distribution(acc['vertical_complexity_proxy'], 'crash records BEFORE')
        flag_distribution(acc_corr, 'crash records AFTER')
        acc_stats = (acc['vertical_complexity_proxy'], acc_corr)
    else:
        log(f"{CRASH_RECORDS} not found; this block is skipped.")

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("G. FIGURES")
    log("=" * 70)

    # Before/after histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    lim = np.percentile(np.abs(proxy), 99)
    for ax, series, title in [
        (axes[0], proxy, 'Original proxy (glo30 - nasadem)'),
        (axes[1], seg['proxy_corr'], f'Corrected proxy (offset {offset:.2f} m removed)')
    ]:
        ax.hist(series.clip(-lim, lim), bins=80, color='steelblue', edgecolor='none')
        ax.axvline(0, color='k', lw=1)
        ax.axvline(series.median(), color='red', ls='--', lw=1,
                   label=f'median = {series.median():.2f} m')
        ax.set_xlabel('vertical_complexity_proxy (m)')
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel('Segments')
    fig.tight_layout()
    fig.savefig('fig_histograma_proxy.png', dpi=150)
    plt.close(fig)
    log("fig_histograma_proxy.png saved")

    # Per-zone before/after boxplot
    names = [n for n in ZONES if len(zones_before[n]) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, data, title in [
        (axes[0], zones_before, 'Before correction'),
        (axes[1], zones_after, 'After correction')
    ]:
        ax.boxplot([data[n].clip(-20, 20) for n in names],
                   tick_labels=[n.replace('_', '\n') for n in names], showfliers=False)
        ax.axhline(0, color='k', lw=1)
        ax.set_ylabel('proxy (m)')
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig('fig_boxplot_zonas.png', dpi=150)
    plt.close(fig)
    log("fig_boxplot_zonas.png saved")

    # -----------------------------------------------------------------
    log("\n" + "=" * 70)
    log("DECISION SUMMARY")
    log("=" * 70)
    log(f"1. Estimated offset: flat+smooth={off_flat:.2f} m, Periferico={off_perif:.2f} m, "
        f"global={off_global:.2f} m")
    log(f"2. Offset adopted in the test: {offset:.2f} m")
    log("3. Review the forest-vs-corridor contrast (section E) and the block-mean")
    log("   range (section C) to decide the correction strategy.")
    log(f"\nReport saved to: {REPORT}")

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LINES))

if __name__ == '__main__':
    main()
