"""
Figure 3 - Reference-zone validation of the dual-DEM proxy.

Creates a publication-quality figure (300 dpi PNG + vector PDF) with:
  (a) a map of the GLO-30 - NASADEM difference over an extended window that
      includes the interior of the La Primavera protected area;
  (b) pixel-level distributions by reference zone.

Run in the same directory as the v2.2 pipeline outputs:
  python figura3_validacion_zonas.py

Required inputs (full reprojected rasters, not the clips):
  - copernicus_gma_utm13n_30m.tif
  - nasadem_gma_utm13n_30m.tif

Outputs:
  - figura3_validacion_zonas.png
  - figura3_validacion_zonas.pdf
  - figura3_validacion_zonas_summary.csv
  - figura3_validacion_zonas_log.txt
"""

from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import geometry_mask
from shapely.geometry import box

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
COP_UTM = BASE / "copernicus_gma_utm13n_30m.tif"
NASA_UTM = BASE / "nasadem_gma_utm13n_30m.tif"

OUT_PNG = BASE / "figura3_validacion_zonas.png"
OUT_PDF = BASE / "figura3_validacion_zonas.pdf"
OUT_CSV = BASE / "figura3_validacion_zonas_summary.csv"
OUT_LOG = BASE / "figura3_validacion_zonas_log.txt"

# Zones in EPSG:4326 [lon_min, lat_min, lon_max, lat_max]
ZONES = {
    "Primavera interior": (-103.7000, 20.5333, -103.5830, 20.6333),
    "Primavera core": (-103.6600, 20.5600, -103.6000, 20.6200),
    "Primavera NE fringe": (-103.5400, 20.6800, -103.4667, 20.7333),
    "Historic centre": (-103.3550, 20.6680, -103.3390, 20.6840),
    "Periferico Norte": (-103.3580, 20.7250, -103.3460, 20.7370),
}
CLASE_ESPERADA = {
    "Primavera interior": "forest",
    "Primavera core": "forest",
    "Primavera NE fringe": "forest",
    "Historic centre": "urban control",
    "Periferico Norte": "urban control",
}

# Work window: full protected area + GMA (UTM 13N)
WORK_BOUNDS = (630000, 2250000, 690000, 2315000)

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def fail(msg):
    log(f"ERROR: {msg}")
    OUT_LOG.write_text("\n".join(LOG), encoding="utf-8")
    sys.exit(1)


def main():
    log("=" * 78)
    log("FIGURE 3 - DUAL-DEM PROXY VALIDATION BY REFERENCE ZONE")
    log(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 78)

    for path in [COP_UTM, NASA_UTM]:
        if not path.exists():
            fail(f"{path.name} was not found. Run the v2.2 pipeline first.")

    # Copernicus: work window and reference grid.
    with rasterio.open(COP_UTM) as src:
        win = rasterio.windows.from_bounds(*WORK_BOUNDS, transform=src.transform)
        win = win.round_offsets().round_lengths()
        glo = src.read(1, window=win).astype("float32")
        glo_nd = src.nodata
        if glo_nd is not None:
            glo[glo == glo_nd] = np.nan
        glo[glo <= 0] = np.nan
        ref_crs = src.crs
        ref_transform = src.window_transform(win)
        ref_shape = glo.shape
        res_m = abs(ref_transform.a)

    log(f"Work window: {WORK_BOUNDS} -> {ref_shape[::-1]} px, res={res_m:.1f} m")

    # NASADEM aligned exactly to the Copernicus window/grid.
    with rasterio.open(NASA_UTM) as src:
        nasa = src.read(1).astype("float32")
        nasa_nd = src.nodata
        nasa_crs = src.crs
        nasa_transform = src.transform

    nasa_al = np.full(ref_shape, np.nan, dtype="float32")
    reproject(
        source=nasa,
        destination=nasa_al,
        src_transform=nasa_transform,
        src_crs=nasa_crs,
        src_nodata=nasa_nd,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    nasa_al[nasa_al <= 0] = np.nan

    diff = glo - nasa_al
    valid = np.isfinite(diff)
    if valid.sum() == 0:
        fail("No valid pixels in the GLO-30 - NASADEM difference.")

    dv = diff[valid]
    global_median = float(np.median(dv))
    log(f"Valid pixels: {valid.sum():,}")
    log(f"Global: mean={dv.mean():.2f} m, median={global_median:.2f} m, "
        f"std={dv.std():.2f} m, positive={100 * (dv > 0).mean():.1f}%")

    # Per-zone statistics.
    results = {}
    rows = []
    for name, bb in ZONES.items():
        bb_utm = transform_bounds("EPSG:4326", ref_crs, *bb)
        geom = box(*bb_utm)
        zone_mask = geometry_mask(
            [geom], out_shape=ref_shape, transform=ref_transform, invert=True
        ) & valid
        vals = diff[zone_mask]
        vals = vals[np.isfinite(vals)]
        results[name] = vals
        if len(vals) == 0:
            log(f"  {name}: 0 valid pixels")
            continue
        row = {
            "zone": name,
            "expected_class": CLASE_ESPERADA[name],
            "n_pixels": len(vals),
            "mean_m": float(vals.mean()),
            "median_m": float(np.median(vals)),
            "std_m": float(vals.std()),
            "p05_m": float(np.percentile(vals, 5)),
            "p90_m": float(np.percentile(vals, 90)),
            "p95_m": float(np.percentile(vals, 95)),
            "pct_positive": float(100 * (vals > 0).mean()),
        }
        rows.append(row)
        log(f"  {name}: n={len(vals):,}, median={row['median_m']:.2f} m, "
            f"p90={row['p90_m']:.2f} m, positive={row['pct_positive']:.1f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_CSV, index=False)
    log(f"Summary CSV: {OUT_CSV.name}")

    # ---- Figure ----
    x0 = ref_transform.c
    y1 = ref_transform.f
    x1 = x0 + ref_shape[1] * ref_transform.a
    y0 = y1 + ref_shape[0] * ref_transform.e

    fig = plt.figure(figsize=(11.6, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.12)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])

    # (a) Map
    vista = np.ma.masked_invalid(np.clip(diff, -10, 10))
    im = ax_map.imshow(
        vista,
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
        extent=[x0, x1, y0, y1],
        origin="upper",
        interpolation="nearest",
    )
    for name, bb in ZONES.items():
        zx0, zy0, zx1, zy1 = transform_bounds("EPSG:4326", ref_crs, *bb)
        ax_map.plot(
            [zx0, zx1, zx1, zx0, zx0],
            [zy0, zy0, zy1, zy1, zy0],
            color="black",
            lw=1.0,
        )
        ax_map.text(
            zx0,
            zy1 + 0.006 * (y1 - y0),
            name,
            fontsize=7.3,
            color="black",
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.68, edgecolor="none", pad=1.0),
        )

    km_formatter = FuncFormatter(lambda v, pos: f"{v / 1000:,.0f}")
    ax_map.xaxis.set_major_formatter(km_formatter)
    ax_map.yaxis.set_major_formatter(km_formatter)
    ax_map.set_xlabel("Easting (km, UTM zone 13N)", fontsize=9)
    ax_map.set_ylabel("Northing (km, UTM zone 13N)", fontsize=9)
    ax_map.tick_params(labelsize=8)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.text(-0.03, 1.03, "a", transform=ax_map.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.040, pad=0.025)
    cbar.set_label("GLO-30 $-$ NASADEM (m)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # (b) Per-zone boxplots
    order = list(ZONES.keys())
    data = [results[n] for n in order]
    colors = ["#2E7D32" if CLASE_ESPERADA[n] == "forest" else "#616161" for n in order]

    bp = ax_box.boxplot(
        data,
        labels=[f"{n}\n(n={len(results[n]):,})" for n in order],
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        widths=0.62,
        medianprops=dict(color="black", linewidth=1.4),
        boxprops=dict(linewidth=0.9),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax_box.axhline(
        global_median,
        color="#7B1FA2",
        linestyle="--",
        linewidth=1.1,
        label=f"Window median ({global_median:.2f} m)",
    )
    ax_box.axhline(0, color="0.35", linestyle=":", linewidth=0.9)
    ax_box.set_ylabel("GLO-30 $-$ NASADEM (m)", fontsize=9)
    ax_box.set_xlabel("Reference zone", fontsize=9)
    ax_box.tick_params(axis="x", labelsize=7.4)
    ax_box.tick_params(axis="y", labelsize=8)
    ax_box.grid(axis="y", color="0.88", linewidth=0.7)
    ax_box.set_axisbelow(True)
    ax_box.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax_box.text(-0.05, 1.03, "b", transform=ax_box.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Saved: {OUT_PNG.name} (300 dpi)")
    log(f"Saved: {OUT_PDF.name} (vector)")
    OUT_LOG.write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
