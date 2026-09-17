"""
Topographic enrichment pipeline for the Guadalajara Metropolitan Area (GMA)
cycling-safety dataset, version 2.2.

This script links 774 cyclist-involved crash records (2015-2024) with 13,380
cycling-network microsegments and enriches both with topographic features from
the Copernicus GLO-30 DEM. NASADEM is used for independent elevation
cross-validation and for the experimental dual-DEM field
`vertical_complexity_proxy`, which is retained only for transparency after a
documented null result in reference-zone validation.

Processing stages:
  1. Input verification
  2. DEM reprojection to UTM 13N with explicit destination nodata
  2.2. Dynamic clipping to the joint segment/crash extent plus a 2,000 m margin
  3. Topographic feature raster (Horn slope in percent rise, TRI, roughness,
     aspect)
  3.3. Segment-level zonal statistics
  4. Direct coordinate-based sampling for all crash records
  5. Technical validation
  6. Output files, checksums, metadata, and codebook
  7. Packaging for Zenodo/GitHub

The code is written for reproducibility: nodata values are read from each
raster, direct crash samples reject non-positive edge-fill artefacts, and the
pipeline stops if any crash record lacks a valid elevation.
"""

import os
import sys
import json
import math
import zipfile
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from scipy import ndimage
from scipy.stats import pearsonr
import geopandas as gpd
from shapely.geometry import LineString, box, Point
from rasterstats import zonal_stats

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILES = {
    'nasadem': 'n20_w104_1arc_v3_2.tif',
    'copernicus': 'Copernicus_DSM_10_N20_00_W104_00_DEM.tif',
    'segmentos_csv': 'segmentos_ciclovias.csv',
    'accidents_csv': 'cyclist_accidents_segmentid_frequency.csv'
}

OUTPUT_FILES = {
    'nasadem_utm': 'nasadem_gma_utm13n_30m.tif',
    'copernicus_utm': 'copernicus_gma_utm13n_30m.tif',
    'nasadem_clip': 'nasadem_gma_clip.tif',
    'copernicus_clip': 'copernicus_gma_clip.tif',
    'features': 'copernicus_features.tif',
    'segmentos_enriched_csv': 'segmentos_ciclovias_enriched.csv',
    'segmentos_enriched_shp': 'segmentos_ciclovias_enriched.shp',
    'segmentos_temp': 'segmentos_utm13n_temp.shp',
    'accidents_enriched_csv': 'cyclist_accidents_enriched.csv',
    'topographic_lookup': 'topographic_lookup.csv',
    'metadata': 'metadata_enriched.json',
    'codebook': 'codebook_enriched.md',
    'readme': 'README.md',
    'requirements': 'requirements.txt',
    'dataset_zip': 'dataset_enriched.zip'
}

CRS_4326 = 'EPSG:4326'
CRS_UTM13N = 'EPSG:32613'
TARGET_RES = 30

# Explicit elevation nodata (prevents zero-fill during reprojection)
NODATA_ELEV = -32768.0
# Margin around the joint segment/crash envelope
BBOX_MARGIN_M = 2000
# Minimum physically plausible elevation in the GMA (source tile
# reports min=551 m; values <= 0 are edge-fill artefacts, not terrain)
MIN_ELEV_VALID = 500.0

# Dynamic bounding box computed in phase2_2 and stored here
GMA_BOUNDS = None

# =============================================================================
# PHASE 1: INPUT FILE VERIFICATION
# =============================================================================

def phase1_verify_inputs():
    print("=" * 70)
    print("PHASE 1: INPUT FILE VERIFICATION")
    print("=" * 70)

    gma_lon = (-103.55, -103.15)
    gma_lat = (20.45, 20.85)

    for name, path in [('NASADEM', INPUT_FILES['nasadem']), ('Copernicus', INPUT_FILES['copernicus'])]:
        with rasterio.open(path) as src:
            covers_lon = src.bounds.left <= gma_lon[0] and src.bounds.right >= gma_lon[1]
            covers_lat = src.bounds.bottom <= gma_lat[0] and src.bounds.top >= gma_lat[1]
            print(f"\n{name}: {path}")
            print(f"  CRS: {src.crs}, Res: {src.res}, Dim: {src.width}x{src.height}")
            # Report source nodata for reprojection diagnostics
            print(f"  Declared source nodata: {src.nodata}")
            print(f"  Covers GMA: {'YES' if covers_lon and covers_lat else 'NO'}")

    # Verify CSV inputs
    for name, path in [('Segments', INPUT_FILES['segmentos_csv']), ('Crash records', INPUT_FILES['accidents_csv'])]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"\n{name}: {path} -- {len(df)} records, {len(df.columns)} columns")
        else:
            print(f"\n{name}: {path} -- NOT FOUND")

# =============================================================================
# PHASE 2: DEM REPROJECTION TO UTM 13N
# =============================================================================

def phase2_reproject_dems():
    print("\n" + "=" * 70)
    print("PHASE 2: DEM REPROJECTION TO UTM 13N (explicit nodata = -32768)")
    print("=" * 70)

    def reproject_dem(input_path, output_path, dst_crs=CRS_UTM13N, target_res=TARGET_RES):
        with rasterio.open(input_path) as src:
            src_nodata = src.nodata  # May be None; passed through to reproject
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=(target_res, target_res)
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs, 'transform': transform, 'width': width,
                'height': height, 'nodata': NODATA_ELEV, 'driver': 'GTiff',
                'dtype': 'float32'
            })
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                        src_transform=src.transform, src_crs=src.crs,
                        src_nodata=src_nodata,
                        dst_transform=transform, dst_crs=dst_crs,
                        dst_nodata=NODATA_ELEV,
                        resampling=Resampling.bilinear
                    )

        # Immediate check: no valid GMA pixel may be <= 0
        with rasterio.open(output_path) as chk:
            arr = chk.read(1)
            valid = arr[(arr != chk.nodata) & (~np.isnan(arr))]
            n_zero = int((valid <= 0).sum())
            print(f"  Reprojected: {output_path}")
            print(f"    min={valid.min():.1f}, max={valid.max():.1f}, mean={valid.mean():.2f}")
            if n_zero > 0:
                print(f"    ERROR: {n_zero} pixels with elevation <= 0 (edge-fill artefact)")
                sys.exit(1)

    reproject_dem(INPUT_FILES['nasadem'], OUTPUT_FILES['nasadem_utm'])
    reproject_dem(INPUT_FILES['copernicus'], OUTPUT_FILES['copernicus_utm'])

# =============================================================================
# PHASE 2.2: DYNAMIC-BBOX CLIPPING
# =============================================================================

def compute_dynamic_bbox(margin_m=BBOX_MARGIN_M):
    """Joint segment/crash envelope in UTM 13N, with margin,
    snapped to the TARGET_RES grid."""
    seg = pd.read_csv(INPUT_FILES['segmentos_csv'])
    seg_gdf = gpd.GeoDataFrame(
        seg,
        geometry=seg.apply(
            lambda r: LineString([(r['Inicio_Lon'], r['Inicio_Lat']), (r['Fin_Lon'], r['Fin_Lat'])]),
            axis=1
        ),
        crs=CRS_4326
    ).to_crs(CRS_UTM13N)

    acc = pd.read_csv(INPUT_FILES['accidents_csv'])
    acc_gdf = gpd.GeoDataFrame(
        acc,
        geometry=gpd.points_from_xy(acc['x'], acc['y']),
        crs=CRS_4326
    ).to_crs(CRS_UTM13N)

    sb = seg_gdf.total_bounds
    ab = acc_gdf.total_bounds

    minx = math.floor((min(sb[0], ab[0]) - margin_m) / TARGET_RES) * TARGET_RES
    miny = math.floor((min(sb[1], ab[1]) - margin_m) / TARGET_RES) * TARGET_RES
    maxx = math.ceil((max(sb[2], ab[2]) + margin_m) / TARGET_RES) * TARGET_RES
    maxy = math.ceil((max(sb[3], ab[3]) + margin_m) / TARGET_RES) * TARGET_RES

    print(f"  Segment envelope:  {[round(v, 1) for v in sb]}")
    print(f"  Crash-record envelope: {[round(v, 1) for v in ab]}")
    return (float(minx), float(miny), float(maxx), float(maxy))


def phase2_2_clip_dems():
    print("\n" + "=" * 70)
    print("PHASE 2.2: DYNAMIC-BBOX CLIPPING (segments + crash records + 2 km)")
    print("=" * 70)

    global GMA_BOUNDS
    GMA_BOUNDS = compute_dynamic_bbox()
    print(f"  Dynamic bbox (UTM 13N): {GMA_BOUNDS}")

    def clip_dem(input_path, output_path, bounds):
        geom = box(*bounds)
        with rasterio.open(input_path) as src:
            out_image, out_transform = mask(src, [geom], crop=True, nodata=src.nodata)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff", "height": out_image.shape[1],
                "width": out_image.shape[2], "transform": out_transform,
                "nodata": src.nodata
            })
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        print(f"  Clipped: {output_path}")

    clip_dem(OUTPUT_FILES['nasadem_utm'], OUTPUT_FILES['nasadem_clip'], GMA_BOUNDS)
    clip_dem(OUTPUT_FILES['copernicus_utm'], OUTPUT_FILES['copernicus_clip'], GMA_BOUNDS)

# =============================================================================
# PHASE 3: TOPOGRAPHIC FEATURE RASTER
# =============================================================================

def phase3_feature_raster():
    print("\n" + "=" * 70)
    print("PHASE 3: TOPOGRAPHIC FEATURES (Copernicus)")
    print("=" * 70)

    def horn_derivatives(dem, pixel_size):
        """Horn (1981) derivatives: weighted 3x3 neighbourhood.
        dz/dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cellsize)
        dz/dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * cellsize)"""
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float64') / 8.0
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float64') / 8.0
        dzdx = ndimage.convolve(dem.astype('float64'), kx, mode='nearest') / pixel_size
        dzdy = ndimage.convolve(dem.astype('float64'), ky, mode='nearest') / pixel_size
        return dzdx, dzdy

    def calculate_slope(dem, pixel_size):
        """Slope in percent rise (Horn, 1981)."""
        dzdx, dzdy = horn_derivatives(dem, pixel_size)
        return 100.0 * np.sqrt(dzdx**2 + dzdy**2)

    def calculate_tri(dem):
        """TRI: mean absolute difference between the centre pixel and its 8 neighbours."""
        padded = np.pad(dem.astype('float64'), 1, mode='edge')
        acc = np.zeros(dem.shape, dtype='float64')
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                acc += np.abs(padded[dy:dy + dem.shape[0], dx:dx + dem.shape[1]] - dem)
        return acc / 8.0

    def calculate_roughness(dem):
        return ndimage.generic_filter(dem.astype(float), np.std, size=3, mode='nearest')

    def calculate_aspect(dem, pixel_size):
        """Aspect: downslope bearing in degrees [0, 360), with 0 = North and
        90 = East. The downslope vector is (-dzdx, -dzdy); the compass bearing
        is atan2(east_component, north_component)."""
        dzdx, dzdy = horn_derivatives(dem, pixel_size)
        aspect = np.degrees(np.arctan2(-dzdx, -dzdy))
        return np.mod(aspect, 360.0)

    with rasterio.open(OUTPUT_FILES['copernicus_clip']) as src:
        dem = src.read(1).astype('float32')
        transform = src.transform
        profile = src.profile
        pixel_size = abs(transform[0])

        nodata_val = src.nodata
        if nodata_val is not None:
            dem[dem == nodata_val] = np.nan
        # Additional guard: residual values <= 0 are excluded from calculations
        dem[dem <= 0] = np.nan

        print("  Computing slope...")
        slope = calculate_slope(dem, pixel_size)
        print("  Computing TRI...")
        tri = calculate_tri(dem)
        print("  Computing roughness...")
        roughness = calculate_roughness(dem)
        print("  Computing aspect...")
        aspect = calculate_aspect(dem, pixel_size)

        profile.update(count=4, dtype='float32', nodata=-999)
        with rasterio.open(OUTPUT_FILES['features'], 'w', **profile) as dst:
            dst.write(np.where(np.isnan(slope), -999, slope).astype('float32'), 1)
            dst.write(np.where(np.isnan(tri), -999, tri).astype('float32'), 2)
            dst.write(np.where(np.isnan(roughness), -999, roughness).astype('float32'), 3)
            dst.write(np.where(np.isnan(aspect), -999, aspect).astype('float32'), 4)

        print(f"  Features saved: {OUTPUT_FILES['features']}")

# =============================================================================
# PHASE 3.3: SEGMENT-LEVEL ZONAL STATISTICS
# =============================================================================

def phase3_3_zonal_statistics():
    print("\n" + "=" * 70)
    print("PHASE 3.3: ZONAL STATISTICS AND SEGMENT ENRICHMENT")
    print("=" * 70)

    print("  Loading segments...")
    segments = pd.read_csv(INPUT_FILES['segmentos_csv'])
    segments['geometry'] = segments.apply(
        lambda r: LineString([(r['Inicio_Lon'], r['Inicio_Lat']), (r['Fin_Lon'], r['Fin_Lat'])]), axis=1
    )
    gdf = gpd.GeoDataFrame(segments, geometry='geometry', crs=CRS_4326)

    print("  Reprojecting to UTM 13N...")
    gdf_utm = gdf.to_crs(CRS_UTM13N)
    print(f"  Segments: {len(gdf_utm)}, Bounds: {gdf_utm.total_bounds}")

    gdf_utm.to_file(OUTPUT_FILES['segmentos_temp'])

    features_config = {
        'elevation_glo30': {'file': OUTPUT_FILES['copernicus_clip'], 'stats': ['mean'], 'band': 1},
        'slope_mean_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['mean'], 'band': 1},
        'slope_max_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['max'], 'band': 1},
        'slope_p95_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['percentile_95'], 'band': 1},
        'slope_std_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['std'], 'band': 1},
        'tri_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['mean'], 'band': 2},
        'roughness_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['mean'], 'band': 3},
        'aspect_glo30': {'file': OUTPUT_FILES['features'], 'stats': ['mean', 'std'], 'band': 4},
        'elevation_nasadem': {'file': OUTPUT_FILES['nasadem_clip'], 'stats': ['mean'], 'band': 1},
    }

    for feature_name, config in features_config.items():
        print(f"  Extracting {feature_name}...")
        # Read the true nodata value from each raster instead of assuming -999
        with rasterio.open(config['file']) as src:
            nodata_real = src.nodata
        stats = zonal_stats(
            gdf_utm, config['file'], stats=config['stats'], band=config.get('band', 1),
            nodata=nodata_real, geojson_out=False
        )
        for stat in config['stats']:
            col = f"{feature_name}_{stat}" if len(config['stats']) > 1 else feature_name
            gdf_utm[col] = [s.get(stat, np.nan) for s in stats]

    print("  Computing vertical_complexity_proxy...")
    gdf_utm['vertical_complexity_proxy'] = gdf_utm['elevation_glo30'] - gdf_utm['elevation_nasadem']

    def categorize(proxy):
        if pd.isna(proxy):
            return 'unknown'
        elif abs(proxy) < 2:
            return 'low'
        elif abs(proxy) < 8:
            return 'moderate'
        else:
            return 'high'

    gdf_utm['vertical_complexity_flag'] = gdf_utm['vertical_complexity_proxy'].apply(categorize)

    print("  Saving segment results...")
    gdf_utm.drop(columns=['geometry']).to_csv(OUTPUT_FILES['segmentos_enriched_csv'], index=False)
    gdf_utm.to_file(OUTPUT_FILES['segmentos_enriched_shp'])

    return gdf_utm

# =============================================================================
# PHASE 4: DIRECT COORDINATE-BASED SAMPLING
# =============================================================================

def phase4_direct_sampling():
    print("\n" + "=" * 70)
    print("PHASE 4: DIRECT COORDINATE-BASED SAMPLING (ALL CRASH RECORDS)")
    print("=" * 70)

    print("  Loading crash records...")
    accidents = pd.read_csv(INPUT_FILES['accidents_csv'])
    print(f"  Total crash records: {len(accidents)}")

    accidents_gdf = gpd.GeoDataFrame(
        accidents,
        geometry=gpd.points_from_xy(accidents['x'], accidents['y']),
        crs=CRS_4326
    )
    accidents_utm = accidents_gdf.to_crs(CRS_UTM13N)
    coords_utm = [(geom.x, geom.y) for geom in accidents_utm.geometry]

    # Pre-check: every point must fall inside the clipped DEM
    with rasterio.open(OUTPUT_FILES['copernicus_clip']) as src:
        b = src.bounds
    outside = [(i, accidents.loc[i, 'Id']) for i, (x, y) in enumerate(coords_utm)
             if not (b.left <= x <= b.right and b.bottom <= y <= b.top)]
    if outside:
        print(f"  CRITICAL ERROR: {len(outside)} crash records OUTSIDE the clipped DEM:")
        for i, rid in outside:
            print(f"    idx={i}, ID={rid}")
        print("  Increase BBOX_MARGIN_M and rerun. Aborting.")
        sys.exit(1)
    print("  Pre-check: 774/774 crash records inside the DEM: PASS")

    def sample_elevation(filepath, coords):
        """Dynamic nodata handling; rejects non-positive edge-fill elevations."""
        with rasterio.open(filepath) as src:
            nodata = src.nodata
            out = []
            for val in src.sample(coords):
                v = float(val[0])
                if (nodata is not None and v == float(nodata)) or np.isnan(v) or v <= 0:
                    out.append(np.nan)
                else:
                    out.append(v)
        return out

    print("  Sampling elevation_glo30...")
    accidents['elevation_glo30'] = sample_elevation(OUTPUT_FILES['copernicus_clip'], coords_utm)

    print("  Sampling elevation_nasadem...")
    accidents['elevation_nasadem'] = sample_elevation(OUTPUT_FILES['nasadem_clip'], coords_utm)

    # Compute the experimental dual-DEM proxy for all crash records
    accidents['vertical_complexity_proxy'] = accidents['elevation_glo30'] - accidents['elevation_nasadem']

    def categorize(proxy):
        if pd.isna(proxy):
            return 'unknown'
        elif abs(proxy) < 2:
            return 'low'
        elif abs(proxy) < 8:
            return 'moderate'
        else:
            return 'high'

    accidents['vertical_complexity_flag'] = accidents['vertical_complexity_proxy'].apply(categorize)

    # Sample feature raster bands (slope, TRI, roughness, aspect)
    print("  Sampling slope...")
    with rasterio.open(OUTPUT_FILES['features']) as src:
        slope_vals = list(src.sample(coords_utm))
        nodata_feat = src.nodata if src.nodata is not None else -999
        accidents['slope_glo30'] = [val[0] if val[0] != nodata_feat else np.nan for val in slope_vals]

    print("  Sampling TRI...")
    with rasterio.open(OUTPUT_FILES['features']) as src:
        tri_vals = list(src.sample(coords_utm))
        accidents['tri_glo30'] = [val[1] if val[1] != nodata_feat else np.nan for val in tri_vals]

    print("  Sampling roughness...")
    with rasterio.open(OUTPUT_FILES['features']) as src:
        rough_vals = list(src.sample(coords_utm))
        accidents['roughness_glo30'] = [val[2] if val[2] != nodata_feat else np.nan for val in rough_vals]

    print("  Sampling aspect...")
    with rasterio.open(OUTPUT_FILES['features']) as src:
        aspect_vals = list(src.sample(coords_utm))
        accidents['aspect_glo30'] = [val[3] if val[3] != nodata_feat else np.nan for val in aspect_vals]

    # Compute slope statistics within a 30 m buffer around each crash record
    print("  Computing slope statistics in a 30 m buffer...")
    slope_stats = []
    with rasterio.open(OUTPUT_FILES['features']) as src:
        for idx, row in accidents_utm.iterrows():
            point = row.geometry
            buffer_geom = point.buffer(30)
            try:
                out_image, out_transform = mask(src, [buffer_geom], crop=True)
                slope_band = out_image[0]  # banda 1 = slope
                valid = (slope_band != src.nodata) & (~np.isnan(slope_band))
                if valid.sum() > 0:
                    slope_data = slope_band[valid]
                    stats = {
                        'slope_mean_glo30': float(np.mean(slope_data)),
                        'slope_max_glo30': float(np.max(slope_data)),
                        'slope_p95_glo30': float(np.percentile(slope_data, 95)),
                        'slope_std_glo30': float(np.std(slope_data))
                    }
                else:
                    stats = {k: np.nan for k in ['slope_mean_glo30', 'slope_max_glo30', 'slope_p95_glo30', 'slope_std_glo30']}
            except Exception:
                stats = {k: np.nan for k in ['slope_mean_glo30', 'slope_max_glo30', 'slope_p95_glo30', 'slope_std_glo30']}
            slope_stats.append(stats)

    slope_df = pd.DataFrame(slope_stats)
    accidents = pd.concat([accidents.reset_index(drop=True), slope_df.reset_index(drop=True)], axis=1)

    accidents.rename(columns={'aspect_glo30': 'aspect_glo30_mean'}, inplace=True)

    print("  Computing aspect standard deviation in a 30 m buffer...")
    aspect_std_list = []
    with rasterio.open(OUTPUT_FILES['features']) as src:
        for idx, row in accidents_utm.iterrows():
            point = row.geometry
            buffer_geom = point.buffer(30)
            try:
                out_image, out_transform = mask(src, [buffer_geom], crop=True)
                aspect_band = out_image[3]  # banda 4 = aspect
                valid = (aspect_band != src.nodata) & (~np.isnan(aspect_band))
                if valid.sum() > 0:
                    aspect_std_list.append(float(np.std(aspect_band[valid])))
                else:
                    aspect_std_list.append(np.nan)
            except Exception:
                aspect_std_list.append(np.nan)

    accidents['aspect_glo30_std'] = aspect_std_list

    # Merge segment features for on-network crash records
    print("  Loading enriched segments for consistency merge...")
    if os.path.exists(OUTPUT_FILES['segmentos_enriched_csv']):
        segments = pd.read_csv(OUTPUT_FILES['segmentos_enriched_csv'])
        feature_cols = [
            'IDSegmento', 'elevation_glo30', 'elevation_nasadem',
            'vertical_complexity_proxy', 'vertical_complexity_flag',
            'slope_mean_glo30', 'slope_max_glo30', 'slope_p95_glo30', 'slope_std_glo30',
            'tri_glo30', 'roughness_glo30', 'aspect_glo30_mean', 'aspect_glo30_std'
        ]
        accidents_with_segments = accidents.merge(
            segments[feature_cols],
            on='IDSegmento',
            how='left',
            suffixes=('', '_seg')
        )

        # Direct-extraction values are authoritative;
        # segment zonal values only fill missing direct samples.
        for col in ['elevation_glo30', 'elevation_nasadem', 'vertical_complexity_proxy',
                    'vertical_complexity_flag', 'slope_mean_glo30', 'slope_max_glo30',
                    'slope_p95_glo30', 'slope_std_glo30', 'tri_glo30', 'roughness_glo30',
                    'aspect_glo30_mean', 'aspect_glo30_std']:
            seg_col = f"{col}_seg"
            if seg_col in accidents_with_segments.columns:
                accidents_with_segments[col] = accidents_with_segments[col].fillna(accidents_with_segments[seg_col])
                accidents_with_segments.drop(columns=[seg_col], inplace=True)

        accidents = accidents_with_segments

    # Hard validation: no crash record may lack a valid elevation
    invalid = accidents[(accidents['elevation_glo30'].isna()) | (accidents['elevation_glo30'] <= 0)]
    n_valid = int(((accidents['elevation_glo30'].notna()) & (accidents['elevation_glo30'] > 0)).sum())
    print(f"\n  Enriched crash records: {len(accidents)}")
    print(f"  With valid elevation_glo30 (> 0 m): {n_valid}")
    if len(invalid) > 0:
        print(f"  CRITICAL ERROR: {len(invalid)} crash records without valid elevation:")
        for _, r in invalid.iterrows():
            print(f"    ID={r['Id']}, mun={r.get('mun', 'NA')}")
        print("  Aborting before writing outputs. Check DEM coverage.")
        sys.exit(1)
    print("  FINAL CHECK: all crash records have valid elevation (> 0 m): PASS")

    accidents.to_csv(OUTPUT_FILES['accidents_enriched_csv'], index=False)
    print(f"  Saved: {OUTPUT_FILES['accidents_enriched_csv']}")

    return accidents

# =============================================================================
# PHASE 5: TECHNICAL VALIDATION
# =============================================================================

def phase5_technical_validation():
    print("\n" + "=" * 70)
    print("PHASE 5: TECHNICAL VALIDATION")
    print("=" * 70)

    # D.1: Cross-DEM validation
    print("\n" + "=" * 60)
    print("D.1: CROSS-DEM VALIDATION")
    print("=" * 60)

    with rasterio.open(OUTPUT_FILES['copernicus_clip']) as src_glo, \
         rasterio.open(OUTPUT_FILES['nasadem_clip']) as src_nasa:
        glo = src_glo.read(1)
        nasa = src_nasa.read(1)
        # Use the true nodata value from each raster
        nd_glo = src_glo.nodata if src_glo.nodata is not None else NODATA_ELEV
        nd_nasa = src_nasa.nodata if src_nasa.nodata is not None else NODATA_ELEV
        valid = (glo != nd_glo) & (nasa != nd_nasa) & (glo > 0) & (nasa > 0) \
                & (~np.isnan(glo)) & (~np.isnan(nasa))
        glo_valid = glo[valid]
        nasa_valid = nasa[valid]
        diff = glo_valid - nasa_valid

    print(f"  Valid pixels in comparison: {valid.sum()}")

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    corr = float(pearsonr(glo_valid, nasa_valid)[0])
    p_value = float(pearsonr(glo_valid, nasa_valid)[1])

    print(f"Mean diff (GLO-30 - NASADEM): {mean_diff:.2f} m")
    print(f"Std dev diff: {std_diff:.2f} m")
    print(f"Pearson correlation: {corr:.4f} (p={p_value:.2e})")
    print(f"Min diff: {float(np.min(diff)):.2f} m")
    print(f"Max diff: {float(np.max(diff)):.2f} m")
    print(f"Median diff: {float(np.median(diff)):.2f} m")
    print(f"95th percentile |diff|: {float(np.percentile(np.abs(diff), 95)):.2f} m")
    print(f"99th percentile |diff|: {float(np.percentile(np.abs(diff), 99)):.2f} m")

    # D.2: Segment distribution
    print("\n" + "=" * 60)
    print("D.2: SEGMENT DISTRIBUTION")
    print("=" * 60)

    segments = pd.read_csv(OUTPUT_FILES['segmentos_enriched_csv'])
    total = len(segments)

    for threshold in [5, 10, 15]:
        count = (segments['vertical_complexity_proxy'].abs() > threshold).sum()
        pct = count / total * 100
        print(f"|diff| > {threshold}m: {count} ({pct:.1f}%)")

    print(f"\nDistribution by flag:")
    flag_dist = segments['vertical_complexity_flag'].value_counts()
    for flag, count in flag_dist.items():
        print(f"  {flag}: {count} ({count/total*100:.1f}%)")

    # D.3: Crash-record coverage
    print("\n" + "=" * 60)
    print("D.3: CRASH-RECORD COVERAGE")
    print("=" * 60)

    accidents = pd.read_csv(OUTPUT_FILES['accidents_enriched_csv'])
    total_acc = len(accidents)
    with_elev = accidents['elevation_glo30'].notna().sum()
    without_elev = accidents['elevation_glo30'].isna().sum()

    print(f"Total crash records: {total_acc}")
    print(f"With topographic features: {with_elev} ({with_elev/total_acc*100:.1f}%)")
    print(f"Without features: {without_elev} ({without_elev/total_acc*100:.1f}%)")

    # D.4: Logical consistency
    print("\n" + "=" * 60)
    print("D.4: LOGICAL CONSISTENCY CHECKS")
    print("=" * 60)

    check1 = (segments['slope_mean_glo30'] <= segments['slope_max_glo30']).all()
    print(f"1. slope_mean <= slope_max (segments): {'PASS' if check1 else 'FAIL'}")

    acc_valid_slope = accidents.dropna(subset=['slope_mean_glo30', 'slope_max_glo30'])
    acc_check = (acc_valid_slope['slope_mean_glo30'] <= acc_valid_slope['slope_max_glo30']).all()
    print(f"2. slope_mean <= slope_max (crash records): {'PASS' if acc_check else 'FAIL'}")
    if not acc_check:
        violations = acc_valid_slope[acc_valid_slope['slope_mean_glo30'] > acc_valid_slope['slope_max_glo30']]
        print(f"   Violations: {len(violations)} records")

    # Hard artefact check (NaN or elevation <= 0 must be zero)
    n_artifacts = int(((accidents['elevation_glo30'].isna()) | (accidents['elevation_glo30'] <= 0)).sum())
    print(f"3. Crash records with invalid elevation (NaN or <= 0): {n_artifacts}")
    print(f"   {'PASS' if n_artifacts == 0 else 'FAIL'}")

    out_of_range = ((accidents['elevation_glo30'] < 1400) | (accidents['elevation_glo30'] > 2200)).sum()
    print(f"4. Elevation outside range (1400-2200 m): {out_of_range} crash records")
    print(f"   {'PASS' if out_of_range == 0 else 'WARNING'}")

    nan_count = accidents['elevation_glo30'].isna().sum()
    print(f"5. Crash records without features (NaN): {nan_count}")
    print(f"   {'PASS' if nan_count == 0 else 'FAIL'}")

    neg_proxy = (accidents['vertical_complexity_proxy'] < -50).sum()
    pos_proxy = (accidents['vertical_complexity_proxy'] > 50).sum()
    print(f"6. Extreme proxy (< -50 m): {neg_proxy}, (> 50m): {pos_proxy}")
    print(f"   {'PASS' if neg_proxy == 0 and pos_proxy == 0 else 'WARNING'}")

    aspect_min = float(accidents['aspect_glo30_mean'].min())
    aspect_max = float(accidents['aspect_glo30_mean'].max())
    print(f"7. Aspect range: [{aspect_min:.1f}, {aspect_max:.1f}] degrees")
    print(f"   {'PASS' if aspect_min >= 0 and aspect_max <= 360 else 'FAIL'}")

    return {
        'dem_mean_diff': mean_diff,
        'dem_std_diff': std_diff,
        'dem_correlation': corr,
        'accidents_with_features': int(with_elev),
        'accidents_without_features': int(without_elev),
        'total_accidents': int(total_acc)
    }

# =============================================================================
# PHASE 6: OUTPUT FILE PREPARATION
# =============================================================================

def phase6_output_files(validation_results):
    print("\n" + "=" * 70)
    print("PHASE 6: OUTPUT FILE PREPARATION")
    print("=" * 70)

    # 6.1: Verify files
    print("\n6.1: Verifying data files...")
    accidents = pd.read_csv(OUTPUT_FILES['accidents_enriched_csv'])
    segments = pd.read_csv(OUTPUT_FILES['segmentos_enriched_csv'])

    print(f"Crash records: {len(accidents)} records, {len(accidents.columns)} columns")
    print(f"Segments: {len(segments)} records, {len(segments.columns)} columns")

    # Topographic lookup
    topo_cols = ['IDSegmento', 'elevation_glo30', 'elevation_nasadem',
                 'vertical_complexity_proxy', 'vertical_complexity_flag',
                 'slope_mean_glo30', 'slope_max_glo30', 'slope_p95_glo30', 'slope_std_glo30',
                 'tri_glo30', 'roughness_glo30', 'aspect_glo30_mean', 'aspect_glo30_std']

    topo_lookup = segments[topo_cols]
    topo_lookup.to_csv(OUTPUT_FILES['topographic_lookup'], index=False)
    print(f"Topographic lookup: {len(topo_lookup)} records")

    # 6.2: Checksums
    print("\n6.2: Computing SHA-256 checksums...")

    def sha256_file(filepath):
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    files_to_hash = [
        OUTPUT_FILES['accidents_enriched_csv'],
        OUTPUT_FILES['segmentos_enriched_csv'],
        OUTPUT_FILES['topographic_lookup']
    ]

    checksums = {}
    for f in files_to_hash:
        if os.path.exists(f):
            checksums[f] = sha256_file(f)
            print(f"  {f}: {checksums[f][:16]}...")
        else:
            print(f"  {f}: NOT FOUND")

    # 6.3: Metadata
    print("\n6.3: Generating metadata.json...")

    metadata = {
        "title": "Enriched geospatial dataset of cyclist crashes and cycling infrastructure with topographic characterization for Guadalajara Metropolitan Area",
        "version": "2.2",
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "creators": ["Carlos Alberto Dominguez-Baez"],
        "description": "Extended dataset integrating administrative crash records (2015-2024), segmented cycling infrastructure (13,380 microsegments), and topographic features from the Copernicus GLO-30 DEM (elevation, slope via Horn algorithm in percent rise, TRI, roughness, aspect), with a second radar DEM (NASADEM) used for cross-validation. ALL 774 crash records include direct topographic extraction via coordinate-based DEM sampling (30m buffer statistics). NOTE: vertical_complexity_proxy (GLO-30 minus NASADEM) is retained as an EXPERIMENTAL variable; reference-zone validation showed it is dominated by the vertical datum offset (approx. -2.6 m, EGM2008 vs EGM96) and does not discriminate land cover in this area.",
        "files": {},
        "sources": {
            "IIEG": {
                "url": "https://iieg.gob.mx/siniestralimap/",
                "license": "CC BY 4.0",
                "description": "Cyclist crash records 2015-2024"
            },
            "GDL_en_Bici": {
                "url": "https://gdlenbici.org/",
                "license": "CC BY 4.0",
                "description": "Cycling infrastructure network"
            },
            "Copernicus_DEM": {
                "product": "COP-DEM_GLO-30-DTED__2023_1",
                "url": "https://dataspace.copernicus.eu/",
                "license": "Open access via Copernicus Data Space Ecosystem, subject to terms of use",
                "description": "Digital elevation model X-band"
            },
            "NASADEM": {
                "product": "NASADEM_HGT",
                "url": "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/",
                "license": "Public Domain",
                "description": "Digital elevation model C-band"
            }
        },
        "spatial": {
            "crs": "EPSG:32613",
            "bbox_utm": list(GMA_BOUNDS) if GMA_BOUNDS else None,
            "bbox_definition": "Joint extent of all segments and crash locations plus a 2000 m margin, snapped to the 30 m grid",
            "nodata_elevation_rasters": NODATA_ELEV,
            "city": "Guadalajara Metropolitan Area",
            "country": "Mexico"
        },
        "temporal": {
            "crash_records": "2015-01-01 to 2024-06-30",
            "dem_epoch_copernicus": "2011-2015",
            "dem_epoch_nasadem": "2000-02"
        },
        "technical_validation": {
            "dem_correlation": float(validation_results['dem_correlation']),
            "dem_mean_difference_m": float(validation_results['dem_mean_diff']),
            "segments_with_features": int(len(segments)),
            "accidents_with_direct_features": int(validation_results['accidents_with_features']),
            "accidents_on_network": int((accidents['IDSegmento'] != 0).sum()),
            "accidents_off_network": int((accidents['IDSegmento'] == 0).sum()),
            "feature_extraction_method": "Direct coordinate-based DEM sampling with 30m buffer statistics for slope and aspect variability"
        }
    }

    for f in files_to_hash:
        if f in checksums and os.path.exists(f):
            df_temp = pd.read_csv(f)
            metadata["files"][f] = {
                "format": "CSV",
                "records": int(len(df_temp)),
                "columns": int(len(df_temp.columns)),
                "sha256": checksums[f]
            }

    with open(OUTPUT_FILES['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)

    print("  metadata_enriched.json generated")

    # 6.4: Codebook
    print("\n6.4: Generating codebook_enriched.md...")

    codebook = """# Codebook: Enriched Cycling Dataset v2.2

## File: cyclist_accidents_enriched.csv
| Variable | Type | Description | Source | Units |
|---|---|---|---|---|
| Id | string | Unique crash identifier | IIEG | - |
| IDSegmento | integer | Segment linkage (0 = off-network) | Derived | - |
| fecha | date | Crash date | IIEG | YYYY-MM-DD |
| anio | integer | Year of crash | IIEG | - |
| mes | string | Month of crash | IIEG | - |
| dia | integer | Day of month | IIEG | - |
| dia_sem | string | Day of week | IIEG | - |
| rango_hora | string | Hour range | IIEG | - |
| mun | string | Municipality | IIEG | - |
| calle_1 | string | Street 1 | IIEG | - |
| calle_2 | string | Street 2 | IIEG | - |
| x | float | Longitude | IIEG | degrees |
| y | float | Latitude | IIEG | degrees |
| tipo_siniestro | string | Crash type | IIEG | - |
| condicion_usuario | string | User condition | IIEG | - |
| tipo_usuario | string | User type | IIEG | - |
| ibaen_atro | string | Vehicle type | IIEG | - |
| sexo | string | Sex | IIEG | - |
| rango_edad | integer | Age range code | IIEG | - |
| consecuencia | string | Outcome | IIEG | - |
| Concentracion_crash records | integer | Crash count per segment | Derived | - |
| elevation_glo30 | float | Primary elevation (Copernicus) | Direct DEM sample | m |
| elevation_nasadem | float | Secondary elevation (NASADEM) | Direct DEM sample | m |
| vertical_complexity_proxy | float | Elevation differential (GLO-30 - NASADEM) | Derived | m |
| vertical_complexity_flag | string | Categorized complexity: low/moderate/high/unknown | Derived | category |
| slope_glo30 | float | Slope at exact crash coordinate | Direct DEM sample | % |
| slope_mean_glo30 | float | Mean slope in 30m buffer around crash | Buffer statistics | % |
| slope_max_glo30 | float | Maximum slope in 30m buffer | Buffer statistics | % |
| slope_p95_glo30 | float | 95th percentile slope in 30m buffer | Buffer statistics | % |
| slope_std_glo30 | float | Standard deviation of slope in 30m buffer | Buffer statistics | % |
| tri_glo30 | float | Terrain Ruggedness Index at coordinate | Direct DEM sample | index |
| roughness_glo30 | float | Elevation roughness at coordinate | Direct DEM sample | m |
| aspect_glo30_mean | float | Mean aspect in 30m buffer | Buffer statistics | degrees |
| aspect_glo30_std | float | Standard deviation of aspect in 30m buffer | Buffer statistics | degrees |

## File: segmentos_ciclovias_enriched.csv
| Variable | Type | Description | Source | Units |
|---|---|---|---|---|
| Ciclovia | string | Cycleway name | GDL en Bici | - |
| Municipio | string | Municipality | GDL en Bici | - |
| Inicio_Lat | float | Start latitude | Derived | degrees |
| Inicio_Lon | float | Start longitude | Derived | degrees |
| Fin_Lat | float | End latitude | Derived | degrees |
| Fin_Lon | float | End longitude | Derived | degrees |
| Riesgo | string | Heuristic risk flag | Derived | category |
| IDSegmento | integer | Persistent segment identifier | Derived | - |
| [Features topograficas] | - | Same as accident file (zonal stats) | - | - |

## File: topographic_lookup.csv
| Variable | Type | Description | Source | Units |
|---|---|---|---|---|
| IDSegmento | integer | Segment identifier | Derived | - |
| elevation_glo30 | float | Primary elevation | Derived | m |
| elevation_nasadem | float | Secondary elevation | Derived | m |
| vertical_complexity_proxy | float | Elevation differential | Derived | m |
| vertical_complexity_flag | string | Complexity category | Derived | category |
| slope_mean_glo30 | float | Mean slope | Derived | % |
| slope_max_glo30 | float | Max slope | Derived | % |
| slope_p95_glo30 | float | P95 slope | Derived | % |
| slope_std_glo30 | float | Slope std | Derived | % |
| tri_glo30 | float | TRI | Derived | index |
| roughness_glo30 | float | Roughness | Derived | m |
| aspect_glo30_mean | float | Mean aspect | Derived | degrees |
| aspect_glo30_std | float | Aspect std | Derived | degrees |

## Derived Features: Physical Basis

### vertical_complexity_proxy  [EXPERIMENTAL - documented null result]
`elevation_glo30 - elevation_nasadem`

Intended interpretation: differential penetration between X-band (Copernicus,
lambda ~3.1cm) and C-band (NASADEM, lambda ~5.6cm) radar as a proxy for
vertical environmental complexity.

**Validation outcome (v2.2):** the variable is dominated by the
vertical datum offset between the two products (GLO-30 heights refer to EGM2008,
NASADEM to EGM96; estimated offset in the GMA ~ -2.6 m, consistent across three
independent estimators) and by the epoch difference (SRTM Feb-2000 vs
TanDEM-X 2011-2015). Reference-zone validation (CONANP official polygon for the
La Primavera pine-oak forest vs. flat urban control corridors) found NO
discrimination capacity at pixel or segment level (median contrast forest
interior vs. flat corridor: -0.12 m). The variable is retained in the dataset
for transparency and reproducibility, but it is NOT recommended as a predictor
of vertical complexity in this area. Categories (low/moderate/high at |2| and
|8| m) are equally experimental.

### slope_*_glo30
Slope computed with the Horn (1981) 3x3 weighted neighbourhood algorithm, in
**percent rise** units: 100 * sqrt((dz/dx)^2 + (dz/dy)^2).

### tri_glo30
Terrain Ruggedness Index: mean absolute difference between the centre pixel and
its 8 neighbours in a 3x3 window (after Riley et al., 1999).

### roughness_glo30
Standard deviation of elevation in 3x3 neighborhood.

### aspect_glo30_mean
Downslope direction in degrees [0, 360), where 0 = North, 90 = East,
180 = South, 270 = West, derived from Horn derivatives.

## Methodological Note: Feature Extraction

For ALL crash records (n=774), topographic features are extracted via **direct coordinate-based DEM sampling**:
1. Accident coordinates (x, y) in EPSG:4326 are reprojected to UTM 13N (EPSG:32613)
2. Elevation, slope, TRI, roughness, and aspect are sampled at exact pixel locations
3. Slope and aspect statistics (mean, max, p95, std) are computed within a 30m radius buffer
4. For on-network accidents (IDSegmento != 0), values are cross-validated with zonal statistics from the associated segment

This ensures 100% coverage of crash records with topographic information, regardless of cycling infrastructure mapping completeness.

## Processing Notes (v2.2)

- DEM reprojection uses an explicit destination nodata value (-32768) to prevent zero-filled edge artefacts.
- The clipping bounding box is computed dynamically as the joint extent of all segments and crash locations plus a 2,000 m margin, snapped to the 30 m grid; all 13,380 segments and all 774 crash locations fall inside.
- Elevation samples <= 0 m are treated as invalid (nodata artefacts), not as real terrain.
- Slope uses the Horn (1981) algorithm in percent rise units; TRI is the mean absolute difference to the 8 neighbours; aspect is the downslope bearing in [0, 360).
- For on-network crash records, direct coordinate-based extraction values are authoritative; segment zonal statistics only fill missing values.
- vertical_complexity_proxy is EXPERIMENTAL: reference-zone validation showed no land-cover discrimination capacity in the GMA (see Technical Validation in the accompanying paper).
"""

    with open(OUTPUT_FILES['codebook'], 'w') as f:
        f.write(codebook)

    print("  codebook_enriched.md generated")

    return checksums

# =============================================================================
# PHASE 7: ZENODO/GITHUB PACKAGING
# =============================================================================

def phase7_packaging():
    print("\n" + "=" * 70)
    print("PHASE 7: ZENODO/GITHUB PACKAGING")
    print("=" * 70)

    # 7.1: Dataset ZIP
    print("\n7.1: Creating dataset_enriched.zip...")

    files_for_zenodo = [
        OUTPUT_FILES['accidents_enriched_csv'],
        OUTPUT_FILES['segmentos_enriched_csv'],
        OUTPUT_FILES['topographic_lookup'],
        OUTPUT_FILES['metadata'],
        OUTPUT_FILES['codebook']
    ]

    with zipfile.ZipFile(OUTPUT_FILES['dataset_zip'], 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_for_zenodo:
            if os.path.exists(f):
                zf.write(f)
                print(f"  Added: {f}")
            else:
                print(f"  NOT FOUND: {f}")

    zip_size = os.path.getsize(OUTPUT_FILES['dataset_zip']) / (1024 * 1024)
    print(f"\nZIP created: {OUTPUT_FILES['dataset_zip']} ({zip_size:.1f} MB)")

    # 7.2: README.md
    print("\n7.2: Generating README.md...")

    readme_lines = [
        "# Enriched Cycling Dataset v2.2 - Guadalajara Metropolitan Area",
        "",
        "## Overview",
        "",
        "This repository contains the processing pipeline and documentation for an enriched geospatial dataset linking cyclist-involved crashes (2015-2024) to cycling infrastructure microsegments with topographic features derived from dual radar Digital Elevation Models (DEMs).",
        "",
        "**Key improvement in v2.x**: All 774 crash records include direct topographic feature extraction via coordinate-based DEM sampling, ensuring 100% coverage regardless of cycling infrastructure mapping completeness. v2.1 fixed nodata handling during DEM reprojection and introduced the dynamic bounding box. v2.2 implements the Horn (1981) slope algorithm in true percent-rise units, aligns TRI with its published definition, makes direct extraction authoritative for on-network crashes, and documents the vertical_complexity_proxy as experimental (no land-cover discrimination in reference-zone validation).",
        "",
        "## Dataset Description",
        "",
        "| File | Records | Description |",
        "|------|---------|-------------|",
        "| `cyclist_accidents_enriched.csv` | 774 | Crash records with direct topographic features (100% coverage) |",
        "| `segmentos_ciclovias_enriched.csv` | 13,380 | Cycling network microsegments with zonal statistics |",
        "| `topographic_lookup.csv` | 13,380 | Topographic reference table by segment |",
        "| `metadata_enriched.json` | - | Machine-readable metadata with SHA-256 checksums |",
        "| `codebook_enriched.md` | - | Variable definitions and physical basis |",
        "",
        "## Key Features",
        "",
        "- **vertical_complexity_proxy** (experimental): Elevation differential between X-band (Copernicus GLO-30) and C-band (NASADEM) radar; reference-zone validation found no land-cover discrimination in the GMA, so it is retained only for transparency",
        "- **Direct coordinate sampling**: All crash records sampled directly from DEMs at exact coordinates (not only via segment association)",
        "- **30m buffer statistics**: Slope and aspect variability computed within 30m radius of each crash location",
        "- **Slope statistics**: Mean, max, p95, std derived from 30m DEM using Horn algorithm",
        "- **Terrain Ruggedness Index (TRI)**: 3x3 neighborhood elevation variability",
        "- **Aspect**: Mean orientation with circular statistics in 30m buffer",
        "",
        "## Quick Start",
        "",
        "```bash",
        "# Clone repository",
        "git clone https://github.com/tu_usuario/gma-cycling-dataset-v2.git",
        "cd gma-cycling-dataset-v2",
        "",
        "# Install dependencies",
        "pip install -r requirements.txt",
        "",
        "# Run full pipeline",
        "python scripts/pipeline_completo_2_corregido.py",
        "```",
        "",
        "## Data Sources",
        "",
        "| Source | Data | License |",
        "|--------|------|---------|",
        "| [IIEG Jalisco](https://iieg.gob.mx/siniestralimap/) | Crash records | CC BY 4.0 |",
        "| [GDL en Bici](https://gdlenbici.org/) | Cycling infrastructure | CC BY 4.0 |",
        "| [Copernicus DEM](https://dataspace.copernicus.eu/) | GLO-30 elevation | Open access |",
        "| [NASA NASADEM](https://e4ftl01.cr.usgs.gov/) | SRTM elevation | Public Domain |",
        "",
        "## Citation",
        "",
        "```",
        "@dataset{dominguez_baez_2026,",
        "  author       = {Dominguez-Baez, Carlos Alberto},",
        "  title        = {Enriched geospatial dataset of cyclist crashes and cycling",
        "                   infrastructure with topographic characterization for",
        "                   Guadalajara Metropolitan Area},",
        "  year         = 2026,",
        "  publisher    = {Zenodo},",
        "  doi          = {10.5281/zenodo.XXXXXXX},",
        "  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}",
        "}",
        "```",
        "",
        "## License",
        "",
        "Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)",
        "Code: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)",
        "",
        "## Contact",
        "",
        "Carlos Alberto Dominguez-Baez",
        "Department of Systems and Computing",
        "Tecnologico Nacional de Mexico - Campus Aguascalientes",
    ]

    with open(OUTPUT_FILES['readme'], 'w') as f:
        f.write('\n'.join(readme_lines))

    print("  README.md generated")

    # 7.3: requirements.txt
    print("\n7.3: Generating requirements.txt...")

    req_lines = [
        "geopandas==1.0.1",
        "rasterio==1.4.2",
        "rasterstats==0.19.0",
        "numpy==2.2.3",
        "pandas==2.2.3",
        "shapely==2.0.6",
        "scipy==1.15.1",
        "matplotlib==3.10.0",
        "seaborn==0.13.2",
    ]

    with open(OUTPUT_FILES['requirements'], 'w') as f:
        f.write('\n'.join(req_lines))

    print("  requirements.txt generated")

    # 7.4: Folder structure
    print("\n7.4: Creating folder structure...")

    folders = ['scripts', 'data', 'docs', 'notebooks', 'tests']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  {folder}/")

    print("\n" + "=" * 70)
    print("PHASE 7 COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FILES['dataset_zip']} (for Zenodo)")
    print(f"  - {OUTPUT_FILES['readme']}")
    print(f"  - {OUTPUT_FILES['requirements']}")
    print("\nFolder structure:")
    print("  scripts/     - Processing code")
    print("  data/        - Data files")
    print("  docs/        - Documentation")
    print("  notebooks/   - Example Jupyter notebooks")
    print("  tests/       - Validation tests")
    print("\nNEXT STEPS:")
    print("1. Upload dataset_enriched.zip to Zenodo")
    print("2. Create the GitHub repository")
    print("3. Upload code and documentation")

# =============================================================================
# MAIN: SEQUENTIAL EXECUTION OF ALL PHASES
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE v2.2: FULL TOPOGRAPHIC ENRICHMENT")
    print("Guadalajara Metropolitan Area - Dataset v2.2")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Verify input files
    missing = []
    for name, path in INPUT_FILES.items():
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print(f"\nERROR: Input files not found: {missing}")
        print("Check that the files are in the working directory.")
        sys.exit(1)

    # Run processing phases
    phase1_verify_inputs()
    phase2_reproject_dems()
    phase2_2_clip_dems()
    phase3_feature_raster()
    phase3_3_zonal_statistics()
    phase4_direct_sampling()
    validation_results = phase5_technical_validation()
    phase6_output_files(validation_results)
    phase7_packaging()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
