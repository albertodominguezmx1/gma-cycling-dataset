"""
Pixel-level test v2: GLO-30 - NASADEM including the forest interior.

Difference from v1: this script uses the full reprojected rasters
(nasadem_gma_utm13n_30m.tif / copernicus_gma_utm13n_30m.tif), not the clipped
products, so it can reach the interior of the La Primavera protected area
(outside the dynamic segment/crash bounding box).

Read-only. Inputs (same directory):
  - copernicus_gma_utm13n_30m.tif
  - nasadem_gma_utm13n_30m.tif
Outputs:
  - prueba_pixel_proxy_v2.txt
  - fig_mapa_diferencia_v2.png
"""

import os
import sys
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import geometry_mask
from shapely.geometry import box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COP_UTM = 'copernicus_gma_utm13n_30m.tif'
NASA_UTM = 'nasadem_gma_utm13n_30m.tif'
REPORT = 'prueba_pixel_proxy_v2.txt'

# Zones in EPSG:4326 [lon_min, lat_min, lon_max, lat_max]
ZONES = {
    # Deep interior of the protected area (south-central official CONANP polygon)
    'Primavera_interior': (-103.7000, 20.5333, -103.5830, 20.6333),
    # Additional core: centre of the protected area
    'Primavera_nucleo': (-103.6600, 20.5600, -103.6000, 20.6200),
    # Northeastern edge (where the cycling network occurs)
    'Primavera_borde_NE': (-103.5400, 20.6800, -103.4667, 20.7333),
    # Urban controls
    'Centro_Historico_GDL': (-103.3550, 20.6680, -103.3390, 20.6840),
    'Periferico_Norte': (-103.3580, 20.7250, -103.3460, 20.7370),
}

# Work window: large enough to cover the full protected area + GMA (UTM 13N)
WORK_BOUNDS = (630000, 2250000, 690000, 2315000)

LINES = []

def log(msg=''):
    print(msg)
    LINES.append(msg)

def main():
    log("=" * 70)
    log("PIXEL TEST v2: GLO-30 - NASADEM (INCLUDING FOREST INTERIOR)")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    for f in [COP_UTM, NASA_UTM]:
        if not os.path.exists(f):
            print(f"ERROR: {f} was not found. Run this script in the pipeline directory.")
            sys.exit(1)

    # --- Work window on Copernicus ---
    with rasterio.open(COP_UTM) as src:
        win = rasterio.windows.from_bounds(*WORK_BOUNDS, transform=src.transform)
        win = win.round_offsets().round_lengths()
        glo = src.read(1, window=win).astype('float32')
        glo_nd = src.nodata
        glo[glo == glo_nd] = np.nan
        glo[glo <= 0] = np.nan
        ref_crs = src.crs
        ref_transform = src.window_transform(win)
        ref_shape = glo.shape
        res_m = abs(ref_transform[0])

    log(f"\nWork window: {WORK_BOUNDS} -> {ref_shape} px, res={res_m:.1f} m")

    # --- NASADEM aligned to the same window/grid ---
    with rasterio.open(NASA_UTM) as src:
        nasa = src.read(1).astype('float32')
        nasa_nd = src.nodata
        nasa_crs = src.crs
        nasa_transform = src.transform

    nasa_al = np.full(ref_shape, np.nan, dtype='float32')
    reproject(
        source=nasa, destination=nasa_al,
        src_transform=nasa_transform, src_crs=nasa_crs, src_nodata=nasa_nd,
        dst_transform=ref_transform, dst_crs=ref_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )
    nasa_al[nasa_al <= 0] = np.nan

    diff = glo - nasa_al
    valid = ~np.isnan(diff)
    dv = diff[valid]
    log(f"Valid pixels: {valid.sum()}")
    log(f"Window global: mean={dv.mean():.2f}, median={np.median(dv):.2f}, "
        f"std={dv.std():.2f}, %positive={100*(dv > 0).mean():.1f}%")

    # --- Per-zone statistics ---
    log("\n" + "=" * 70)
    log("PER-ZONE DIFFERENCE (PIXEL LEVEL)")
    log("=" * 70)
    results = {}
    for name, bb in ZONES.items():
        bb_utm = transform_bounds('EPSG:4326', ref_crs, *bb)
        geom = box(*bb_utm)
        zone_mask = geometry_mask([geom], out_shape=ref_shape,
                                transform=ref_transform, invert=True) & valid
        vals = diff[zone_mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            log(f"  {name}: 0 valid pixels")
            results[name] = None
            continue
        log(f"  {name}: n={len(vals)} px, mean={vals.mean():.2f}, "
            f"median={np.median(vals):.2f}, std={vals.std():.2f}, "
            f"p75={np.percentile(vals, 75):.2f}, p90={np.percentile(vals, 90):.2f}, "
            f"p95={np.percentile(vals, 95):.2f}, %positive={100*(vals > 0).mean():.1f}%")
        results[name] = vals

    # --- Decisive contrast ---
    log("\n" + "=" * 70)
    log("DECISIVE CONTRAST: DEEP FOREST VS URBAN CONTROLS")
    log("=" * 70)
    for forest in ['Primavera_interior', 'Primavera_nucleo', 'Primavera_borde_NE']:
        zb = results.get(forest)
        if zb is None:
            continue
        for control in ['Centro_Historico_GDL', 'Periferico_Norte']:
            zc = results.get(control)
            if zc is None:
                continue
            log(f"  {forest} vs {control}: "
                f"median difference = {np.median(zb) - np.median(zc):+.2f} m, "
                f"mean difference = {zb.mean() - zc.mean():+.2f} m, "
                f"p90 difference = {np.percentile(zb, 90) - np.percentile(zc, 90):+.2f} m")

    log("\n  DECISION RULE:")
    log("  - Median difference >= ~2-3 m in the interior/core: the signal EXISTS;")
    log("    the problem was segment aggregation -> redesign the variable.")
    log("  - Difference ~0 even in the interior: the signal does NOT exist in this")
    log("    area with these sensors -> reformulate the paper (documented null result).")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(9, 8))
    view = np.clip(diff, -10, 10)
    x0 = ref_transform[2]
    y1 = ref_transform[5]
    x1 = x0 + ref_shape[1] * ref_transform[0]
    y0 = y1 + ref_shape[0] * ref_transform[4]
    im = ax.imshow(view, cmap='RdYlGn', extent=[x0, x1, y0, y1])
    fig.colorbar(im, ax=ax, label='GLO-30 - NASADEM (m)', shrink=0.8)
    for name, bb in ZONES.items():
        zx0, zy0, zx1, zy1 = transform_bounds('EPSG:4326', ref_crs, *bb)
        ax.plot([zx0, zx1, zx1, zx0, zx0], [zy0, zy0, zy1, zy1, zy0], 'k-', lw=1.2)
        ax.text(zx0, zy1, name.replace('_', ' '), fontsize=8, color='k',
                va='bottom', ha='left',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
    ax.set_xlabel('Easting UTM 13N (m)')
    ax.set_ylabel('Northing UTM 13N (m)')
    ax.set_title('GLO-30 - NASADEM difference (clipped to +/-10 m), extended window')
    fig.tight_layout()
    fig.savefig('fig_mapa_diferencia_v2.png', dpi=150)
    plt.close(fig)
    log("\nfig_mapa_diferencia_v2.png saved")

    log(f"\nReport saved to: {REPORT}")
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LINES))

if __name__ == '__main__':
    main()
