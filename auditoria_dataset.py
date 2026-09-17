"""
Technical audit for the GMA cycling-safety dataset, version 2.2.

Levels:
  1. Original and reprojected DEM checks
  2. Topographic feature raster checks
  3. Segment-level zonal statistics
  4. Direct crash-record extraction
  5. Reference-zone evaluation with official boundaries
  6. Reproducibility checks (SHA-256 checksums)

The audit treats NaN or non-positive elevations in reprojected/clipped DEMs as
errors, lists every invalid crash record, and reports reference-zone proxy
results as documented evidence following the null validation outcome.
"""

import os
import sys
import json
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PATHS = {
    'nasadem_orig': 'n20_w104_1arc_v3_2.tif',
    'copernicus_orig': 'Copernicus_DSM_10_N20_00_W104_00_DEM.tif',
    'nasadem_utm': 'nasadem_gma_utm13n_30m.tif',
    'copernicus_utm': 'copernicus_gma_utm13n_30m.tif',
    'nasadem_clip': 'nasadem_gma_clip.tif',
    'copernicus_clip': 'copernicus_gma_clip.tif',
    'features': 'copernicus_features.tif',
    'segmentos_csv': 'segmentos_ciclovias_enriched.csv',
    'segmentos_shp': 'segmentos_ciclovias_enriched.shp',
    'accidentes_csv': 'cyclist_accidents_enriched.csv',
    'metadata': 'metadata_enriched.json',
}

MIN_ELEV_VALID = 500.0  # Below this value, elevations are edge-fill artefacts

REPORT = []
WARNINGS = []
ERRORS = []

def log(msg, level='INFO'):
    print(f"[{level}] {msg}")
    REPORT.append(f"[{level}] {msg}")
    if level == 'WARNING':
        WARNINGS.append(msg)
    elif level == 'ERROR':
        ERRORS.append(msg)

def sha256_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

# =============================================================================
# LEVEL 1: ORIGINAL AND REPROJECTED DEM CHECKS
# =============================================================================

def level1_dems():
    log("\n" + "="*70)
    log("LEVEL 1: ORIGINAL AND REPROJECTED DEM CHECKS")
    log("="*70)

    # 1.1 GMA spatial coverage
    gma_lon = (-103.55, -103.15)
    gma_lat = (20.45, 20.85)

    for name, key in [('NASADEM orig', 'nasadem_orig'), ('Copernicus orig', 'copernicus_orig')]:
        with rasterio.open(PATHS[key]) as src:
            covers_lon = src.bounds.left <= gma_lon[0] and src.bounds.right >= gma_lon[1]
            covers_lat = src.bounds.bottom <= gma_lat[0] and src.bounds.top >= gma_lat[1]
            if covers_lon and covers_lat:
                log(f"{name}: FULLY COVERS THE GMA", 'INFO')
            else:
                log(f"{name}: DOES NOT COVER THE GMA. Bounds: {src.bounds}", 'ERROR')

    # 1.2 Elevation histogram: original vs reprojected
    log("\n--- 1.2 Original vs reprojected elevation histogram ---")

    def elevation_stats(path):
        with rasterio.open(path) as src:
            dem = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                dem = dem[dem != nodata]
            dem = dem[~np.isnan(dem)]
            return dem, {
                'min': float(dem.min()),
                'max': float(dem.max()),
                'mean': float(dem.mean()),
                'median': float(np.median(dem)),
            }

    for orig_key, reproj_key, label in [
        ('nasadem_orig', 'nasadem_utm', 'NASADEM'),
        ('copernicus_orig', 'copernicus_utm', 'Copernicus')
    ]:
        if not os.path.exists(PATHS[reproj_key]):
            log(f"{label} reprojected raster not found; skipping comparison", 'WARNING')
            continue

        arr1, s1 = elevation_stats(PATHS[orig_key])
        arr2, s2 = elevation_stats(PATHS[reproj_key])

        log(f"{label} original:  min={s1['min']:.1f}, max={s1['max']:.1f}, mean={s1['mean']:.2f}, median={s1['median']:.2f}")
        log(f"{label} reprojected: min={s2['min']:.1f}, max={s2['max']:.1f}, mean={s2['mean']:.2f}, median={s2['median']:.2f}")

        # Hard check for edge-fill artefacts in the reprojected raster
        n_invalid = int((arr2 <= 0).sum())
        if n_invalid > 0:
            log(f"{label}: {n_invalid} pixels with elevation <= 0 in the reprojected raster: "
                f"edge-fill from missing dst_nodata -> ERROR", 'ERROR')
        else:
            log(f"{label}: 0 pixels with elevation <= 0 in the reprojected raster: OK", 'INFO')

        if abs(s1['mean']) > 0:
            diff_pct = abs(s1['mean'] - s2['mean']) / abs(s1['mean']) * 100
            if diff_pct < 1.0:
                log(f"{label}: Reprojection preserves elevation (diff {diff_pct:.2f}%)", 'INFO')
            else:
                log(f"{label}: Reprojection ALTERS elevation (diff {diff_pct:.2f}%)", 'ERROR')
        else:
            log(f"{label}: Cannot compute difference because mean=0", 'WARNING')

# =============================================================================
# LEVEL 2: FEATURE RASTER CHECKS
# =============================================================================

def level2_features():
    log("\n" + "="*70)
    log("LEVEL 2: FEATURE RASTER CHECKS")
    log("="*70)

    with rasterio.open(PATHS['features']) as src:
        slope = src.read(1)
        slope = slope[slope != src.nodata]
        slope = slope[~np.isnan(slope)]

        log(f"Slope: min={slope.min():.2f}%, max={slope.max():.2f}%, mean={slope.mean():.2f}%")

        if slope.min() < 0:
            log("Slope has negative values (physically impossible)", 'ERROR')
        else:
            log("Slope >= 0: OK", 'INFO')

        # v2.2: con percent rise real (Horn), >100% es legitimo en barrancas
        if slope.max() > 500:
            log(f"Slope max={slope.max():.2f}% > 500%. Physically suspicious; review", 'WARNING')
        elif slope.max() > 100:
            log(f"Slope max={slope.max():.2f}% > 100%: compatible with canyon walls (>45 degrees)", 'INFO')

        aspect = src.read(4)
        aspect = aspect[aspect != src.nodata]
        aspect = aspect[~np.isnan(aspect)]

        log(f"Aspect: min={aspect.min():.2f}, max={aspect.max():.2f}")

        if aspect.min() >= 0 and aspect.max() < 360:
            log("Aspect within [0, 360): OK", 'INFO')
        else:
            log("Aspect OUTSIDE [0, 360)", 'ERROR')

# =============================================================================
# LEVEL 3: SEGMENT CHECKS (ZONAL STATISTICS)
# =============================================================================

def level3_segments():
    log("\n" + "="*70)
    log("LEVEL 3: SEGMENT CHECKS")
    log("="*70)

    segments = pd.read_csv(PATHS['segmentos_csv'])
    log(f"Total segments: {len(segments)}")

    # 3.1 Values within the source-raster range
    log("\n--- 3.1 Value range vs source raster ---")

    with rasterio.open(PATHS['copernicus_clip']) as src:
        dem = src.read(1)
        mask_valid = ~np.isnan(dem)
        if src.nodata is not None:
            mask_valid &= (dem != src.nodata)
        # The clipped DEM itself must not contain edge-fill values
        n_zero = int((dem[mask_valid] <= 0).sum())
        if n_zero > 0:
            log(f"Copernicus clipped DEM contains {n_zero} pixels with elevation <= 0: ERROR", 'ERROR')
        dem_valid = dem[mask_valid & (dem > 0)]
        dem_min, dem_max = float(dem_valid.min()), float(dem_valid.max())
        log(f"Copernicus clipped DEM (valid): min={dem_min:.2f}, max={dem_max:.2f}")

    out_of_range = ((segments['elevation_glo30'] < dem_min) |
                    (segments['elevation_glo30'] > dem_max)).sum()
    if out_of_range == 0:
        log("All segments have elevation within the DEM range: PASS", 'INFO')
    else:
        log(f"{out_of_range} segments with elevation OUTSIDE the DEM range: FAIL", 'ERROR')

    # 3.2 Proxy sign
    log("\n--- 3.2 Sign of vertical_complexity_proxy ---")
    neg = (segments['vertical_complexity_proxy'] < 0).sum()
    pos = (segments['vertical_complexity_proxy'] > 0).sum()
    zero = (segments['vertical_complexity_proxy'] == 0).sum()
    log(f"Negative proxy: {neg} ({neg/len(segments)*100:.1f}%)")
    log(f"Positive proxy: {pos} ({pos/len(segments)*100:.1f}%)")
    log(f"Zero proxy:     {zero} ({zero/len(segments)*100:.1f}%)")

    if neg > len(segments) * 0.8:
        log("Most proxies are negative, consistent with the documented vertical-datum offset "
            "between GLO-30 (EGM2008) and NASADEM (EGM96); informational only.", 'INFO')
    else:
        log("Acceptable sign distribution", 'INFO')

    # 3.3 Consistency: slope_mean <= slope_max
    check = (segments['slope_mean_glo30'] <= segments['slope_max_glo30']).all()
    if check:
        log("slope_mean <= slope_max for ALL segments: PASS", 'INFO')
    else:
        viol = (segments['slope_mean_glo30'] > segments['slope_max_glo30']).sum()
        log(f"slope_mean > slope_max in {viol} segments: FAIL", 'ERROR')

    # 3.4 NaN in features
    nan_count = segments['elevation_glo30'].isna().sum()
    if nan_count == 0:
        log("0 NaN in elevation_glo30: PASS", 'INFO')
    else:
        log(f"{nan_count} NaN in elevation_glo30: FAIL", 'ERROR')

    # 3.5 Flag distribution
    log("\n--- 3.5 Flag distribution ---")
    flags = segments['vertical_complexity_flag'].value_counts()
    for f, c in flags.items():
        log(f"  {f}: {c} ({c/len(segments)*100:.1f}%)")

# =============================================================================
# LEVEL 4: CRASH-RECORD CHECKS (DIRECT SAMPLING)
# =============================================================================

def level4_crashes():
    log("\n" + "="*70)
    log("LEVEL 4: CRASH-RECORD CHECKS")
    log("="*70)

    accidents = pd.read_csv(PATHS['accidentes_csv'])
    log(f"Total crash records: {len(accidents)}")

    # 4.1 100% coverage
    with_elev = accidents['elevation_glo30'].notna().sum()
    without_elev = accidents['elevation_glo30'].isna().sum()
    log(f"With elevation_glo30: {with_elev} ({with_elev/len(accidents)*100:.1f}%)")
    log(f"Without elevation_glo30: {without_elev} ({without_elev/len(accidents)*100:.1f}%)")

    if without_elev == 0:
        log("100% of crash records with features: PASS", 'INFO')
    else:
        log("Crash records WITHOUT features: FAIL", 'ERROR')

    # 4.2 Off-network vs on-network
    off = accidents[accidents['IDSegmento'] == 0]
    on = accidents[accidents['IDSegmento'] != 0]

    log(f"\nOff-network: {len(off)} crash records")
    log(f"  Mean off-network elevation: {off['elevation_glo30'].mean():.2f} m")
    log(f"On-network:  {len(on)} crash records")
    log(f"  Mean on-network elevation:  {on['elevation_glo30'].mean():.2f} m")

    diff_elev = abs(off['elevation_glo30'].mean() - on['elevation_glo30'].mean())
    if diff_elev > 200:
        log(f"Off- vs on-network elevation difference = {diff_elev:.2f} m (>200 m). Review coordinates", 'WARNING')
    else:
        log(f"Off- vs on-network elevation difference = {diff_elev:.2f} m: acceptable", 'INFO')

    # 4.3 Coordinates inside the DEM + elevation validity (zero tolerance)
    log("\n--- 4.3 Coordinates inside the DEM and valid elevation ---")
    gdf = gpd.GeoDataFrame(
        accidents,
        geometry=gpd.points_from_xy(accidents['x'], accidents['y']),
        crs='EPSG:4326'
    ).to_crs('EPSG:32613')

    with rasterio.open(PATHS['copernicus_clip']) as src:
        bounds = src.bounds
        outside_mask = ((gdf.geometry.x < bounds.left) | (gdf.geometry.x > bounds.right) |
                        (gdf.geometry.y < bounds.bottom) | (gdf.geometry.y > bounds.top))
        outside = int(outside_mask.sum())
        if outside == 0:
            log("All crash records inside the DEM: PASS", 'INFO')
        else:
            log(f"{outside} crash records OUTSIDE the DEM: FAIL", 'ERROR')
            for _, row in accidents[outside_mask.values].iterrows():
                log(f"  Outside DEM: ID={row['Id']}, mun={row.get('mun', 'NA')}", 'ERROR')

    # Invalid elevation (NaN or <= 0) in either DEM -> ERROR
    for col in ['elevation_glo30', 'elevation_nasadem']:
        bad = accidents[(accidents[col].isna()) | (accidents[col] <= 0)]
        if len(bad) == 0:
            log(f"{col}: 0 invalid values (NaN or <= 0): PASS", 'INFO')
        else:
            log(f"{col}: {len(bad)} invalid values (NaN or <= 0): FAIL", 'ERROR')
            for _, row in bad.iterrows():
                log(f"  Invalid: ID={row['Id']}, {col}={row[col]}, mun={row.get('mun', 'NA')}", 'ERROR')

    # 4.4 Consistency: slope_mean <= slope_max
    acc_valid = accidents.dropna(subset=['slope_mean_glo30', 'slope_max_glo30'])
    if len(acc_valid) > 0:
        check = (acc_valid['slope_mean_glo30'] <= acc_valid['slope_max_glo30']).all()
        if check:
            log("slope_mean <= slope_max for ALL crash records: PASS", 'INFO')
        else:
            viol = (acc_valid['slope_mean_glo30'] > acc_valid['slope_max_glo30']).sum()
            log(f"slope_mean > slope_max in {viol} crash records: WARNING", 'WARNING')

    # 4.5 Extreme proxies
    neg_ext = (accidents['vertical_complexity_proxy'] < -50).sum()
    pos_ext = (accidents['vertical_complexity_proxy'] > 50).sum()
    if neg_ext == 0 and pos_ext == 0:
        log("0 extreme proxies (>50 m): PASS", 'INFO')
    else:
        log(f"Extreme proxies: {neg_ext} negative, {pos_ext} positive: WARNING", 'WARNING')

    # 4.6 Aspect range
    asp_min = accidents['aspect_glo30_mean'].min()
    asp_max = accidents['aspect_glo30_mean'].max()
    if asp_min >= 0 and asp_max <= 360:
        log(f"Aspect in [{asp_min:.1f}, {asp_max:.1f}]: PASS", 'INFO')
    else:
        log(f"Aspect out of range: FAIL", 'ERROR')

    # 4.7 Elevation outside the plausible GMA range (valid values > 0 only)
    valid_records = accidents[accidents['elevation_glo30'] > 0]
    out_range_mask = (valid_records['elevation_glo30'] < 1400) | (valid_records['elevation_glo30'] > 2200)
    out_range = int(out_range_mask.sum())
    if out_range == 0:
        log("All valid elevations within [1400, 2200] m: PASS", 'INFO')
    else:
        log(f"{out_range} crash records (with elevation > 0) outside [1400, 2200] m: WARNING (manual review)", 'WARNING')
        outliers = valid_records[out_range_mask]
        for idx, row in outliers.iterrows():
            log(f"  Outlier: ID={row['Id']}, elev={row['elevation_glo30']:.2f}, mun={row['mun']}", 'WARNING')

# =============================================================================
# LEVEL 5: REFERENCE-ZONE EVALUATION
# =============================================================================

def level5_reference_zones():
    log("\n" + "="*70)
    log("LEVEL 5: REFERENCE-ZONE EVALUATION")
    log("="*70)
    log("NOTE: Bounding boxes come from official sources (CONANP, SITEUR,")
    log("verified locations). La Primavera uses the official protected-area limits;")
    log("cycling-network segments are concentrated on its northeastern fringe.")
    log("="*70)

    segments = pd.read_csv(PATHS['segmentos_csv'])

    gdf = gpd.GeoDataFrame(
        segments,
        geometry=gpd.points_from_xy((segments['Inicio_Lon'] + segments['Fin_Lon'])/2,
                                    (segments['Inicio_Lat'] + segments['Fin_Lat'])/2),
        crs='EPSG:4326'
    )

    # Bounding boxes in decimal degrees [lon_min, lat_min, lon_max, lat_max]
    # Coordinates updated from official sources:
    #  - La Primavera: official APFF boundary coordinates from the CONANP/SIMEC
    #    management programme (103 28'-103 42' W, 20 32'-20 44' N; 30,500 ha;
    #    Zapopan 54%, Tala 35%, Tlajomulco 11%). Pine-oak forest.
    #  - Historic centre: approximate centre 20.676 N, -103.347 W; low-to-mid-rise
    #    historic buildings.
    #  - Andares/Puerta de Hierro: Andares shopping centre at 20.710037 N,
    #    -103.412140 W; corporate area with towers (Hyatt Regency ~173 m)
    #    interspersed with low-rise areas.
    #  - Periferico Norte: SITEUR Line 1 station at 20 43'52" N, 103 21'08" W
    #    (~20.731 N, -103.352 W); flat road corridor.
    # NOTE: expected_proxy_min/max are physical expectations for the signed proxy.
    # Because the dual-DEM proxy has a documented null validation result and a
    # systematic vertical-datum offset, these checks are informational only and
    # are never treated as pass/fail criteria.
    zones = {
        'Bosque_La_Primavera': {
            'bbox': (-103.7000, 20.5333, -103.4667, 20.7333),
            'expected_flag': 'high',
            'expected_proxy_min': 5.0,
            'desc': 'APFF La Primavera (CONANP): dense pine-oak forest west of the GMA'
        },
        'Centro_Historico_GDL': {
            'bbox': (-103.3550, 20.6680, -103.3390, 20.6840),
            'expected_flag': 'low',
            'expected_proxy_max': 3.0,
            'desc': 'Historic urban centre (centre ~20.676, -103.347), low-to-mid-rise buildings'
        },
        'Zapopan_Andares': {
            'bbox': (-103.4201, 20.7020, -103.4041, 20.7180),
            'expected_flag': 'moderate',
            'expected_proxy_min': 2.0,
            'expected_proxy_max': 10.0,
            'desc': 'Andares/Puerta de Hierro (centre ~20.7100, -103.4121), corporate-residential area with towers'
        },
        'Periferico_Norte': {
            'bbox': (-103.3580, 20.7250, -103.3460, 20.7370),
            'expected_flag': 'low',
            'expected_proxy_max': 3.0,
            'desc': 'Periferico Norte SITEUR L1 station (~20.731, -103.352), flat road corridor'
        }
    }

    for name, info in zones.items():
        lon_min, lat_min, lon_max, lat_max = info['bbox']
        subset = gdf[(gdf.geometry.x >= lon_min) & (gdf.geometry.x <= lon_max) &
                     (gdf.geometry.y >= lat_min) & (gdf.geometry.y <= lat_max)]

        if len(subset) == 0:
            log(f"{name}: 0 segments in bbox {info['bbox']}; zone skipped (no cycling-network coverage).", 'INFO')
            continue

        mean_proxy = subset['vertical_complexity_proxy'].mean()
        median_proxy = subset['vertical_complexity_proxy'].median()
        mode_flag = subset['vertical_complexity_flag'].mode()[0] if len(subset['vertical_complexity_flag'].mode()) > 0 else 'N/A'

        log(f"\n{name}: {len(subset)} segments")
        log(f"  Description: {info['desc']}")
        log(f"  Mean proxy: {mean_proxy:.2f} m, Median: {median_proxy:.2f} m")
        log(f"  Most common flag: {mode_flag}")

        observations = []
        if 'expected_flag' in info:
            observations.append(f"flag={mode_flag} (physical expectation: {info['expected_flag']})")
        if 'expected_proxy_min' in info:
            observations.append(f"proxy mean {mean_proxy:.2f} vs expected min {info['expected_proxy_min']}")
        if 'expected_proxy_max' in info:
            observations.append(f"proxy mean {mean_proxy:.2f} vs expected max {info['expected_proxy_max']}")
        if not observations:
            observations.append("No reference criteria defined")

        for obs in observations:
            log(f"  {obs} [informational only; documented null proxy result]", 'INFO')

# =============================================================================
# LEVEL 6: REPRODUCIBILITY (CHECKSUMS)
# =============================================================================

def level6_reproducibility():
    log("\n" + "="*70)
    log("LEVEL 6: REPRODUCIBILITY CHECK")
    log("="*70)

    output_files = [
        PATHS['accidentes_csv'],
        PATHS['segmentos_csv'],
        'topographic_lookup.csv',
        PATHS['metadata'],
    ]

    if os.path.exists(PATHS['metadata']):
        with open(PATHS['metadata'], 'r') as f:
            meta = json.load(f)

        log("Checksums in metadata.json:")
        for fname, fmeta in meta.get('files', {}).items():
            if os.path.exists(fname):
                actual = sha256_file(fname)
                expected = fmeta.get('sha256', '')
                if actual == expected:
                    log(f"  {fname}: CHECKSUM OK", 'INFO')
                else:
                    log(f"  {fname}: CHECKSUM MISMATCH", 'WARNING')
                    log(f"    Expected: {expected[:16]}...", 'WARNING')
                    log(f"    Actual:   {actual[:16]}...", 'WARNING')
            else:
                log(f"  {fname}: FILE NOT FOUND", 'WARNING')
    else:
        log("metadata.json not found; computing current checksums:", 'INFO')
        for fname in output_files:
            if os.path.exists(fname):
                cs = sha256_file(fname)
                log(f"  {fname}: {cs}")
            else:
                log(f"  {fname}: NOT FOUND", 'WARNING')

# =============================================================================
# FINAL REPORT
# =============================================================================

def final_report():
    log("\n" + "="*70)
    log("FINAL AUDIT REPORT")
    log("="*70)
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total checks: {len(REPORT)}")
    log(f"Errors: {len(ERRORS)}")
    log(f"Warnings: {len(WARNINGS)}")

    if len(ERRORS) == 0:
        log("GLOBAL RESULT: PASS (no critical errors)", 'INFO')
    else:
        log("GLOBAL RESULT: FAIL (critical errors present)", 'ERROR')

    if len(WARNINGS) > 0:
        log("Review warnings before submission.", 'WARNING')

    report_path = 'technical_audit_2.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT))
    log(f"\nReport saved to: {report_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("COMPLETE TECHNICAL AUDIT v2.2 - GMA Dataset")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    missing = []
    for key, path in PATHS.items():
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print(f"\nERROR: Files not found: {missing}")
        print("Run this script in the same directory as the pipeline outputs.")
        sys.exit(1)

    level1_dems()
    level2_features()
    level3_segments()
    level4_crashes()
    level5_reference_zones()
    level6_reproducibility()
    final_report()

    print("\n" + "="*70)
    print("AUDIT COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()
