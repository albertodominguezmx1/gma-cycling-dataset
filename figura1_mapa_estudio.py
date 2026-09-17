"""
Figure 1 - Study-area map (Scientific Data).

Creates a publication-quality figure (300 dpi PNG + vector PDF) with:
  - shaded relief and elevation from the clipped Copernicus GLO-30 DEM;
  - the 13,380 cycling-network microsegments;
  - the 774 cyclist-involved crash records;
  - reference cities, north arrow, scale bar, and legend.

Run in the same directory as the v2.2 pipeline outputs:
  python figura1_mapa_estudio.py

Required inputs:
  - copernicus_gma_clip_v2.tif
  - segmentos_ciclovias_enriched_v2.shp   (or segmentos_ciclovias_enriched_v2.csv)
  - cyclist_accidents_enriched_v2.csv

Outputs:
  - figura1_area_estudio.png
  - figura1_area_estudio.pdf
  - figura1_area_estudio_log.txt
"""

from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parent
DEM_CLIP = BASE / "copernicus_gma_clip_v2.tif"
SEG_SHP = BASE / "segmentos_ciclovias_enriched_v2.shp"
SEG_ENRICHED_CSV = BASE / "segmentos_ciclovias_enriched_v2.csv"
SEG_RAW_CSV = BASE / "segmentos_ciclovias.csv"
ACC_CSV = BASE / "cyclist_accidents_enriched_v2.csv"

OUT_PNG = BASE / "figura1_area_estudio.png"
OUT_PDF = BASE / "figura1_area_estudio.pdf"
OUT_LOG = BASE / "figura1_area_estudio_log.txt"

LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


def fail(msg):
    log(f"ERROR: {msg}")
    (BASE / OUT_LOG.name).write_text("\n".join(LOG), encoding="utf-8")
    sys.exit(1)


def load_segments(target_crs):
    """Load segments from SHP (preferred) or from a WGS84 coordinate CSV."""
    if SEG_SHP.exists():
        gdf = gpd.read_file(SEG_SHP)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:32613")
        return gdf.to_crs(target_crs), str(SEG_SHP.name)

    csv_path = SEG_ENRICHED_CSV if SEG_ENRICHED_CSV.exists() else SEG_RAW_CSV
    if not csv_path.exists():
        fail("segmentos_ciclovias_enriched_v2.shp or a segment CSV was not found.")

    seg = pd.read_csv(csv_path)
    required = {"Inicio_Lat", "Inicio_Lon", "Fin_Lat", "Fin_Lon"}
    if not required.issubset(seg.columns):
        fail(f"El CSV {csv_path.name} no contiene las columnas {sorted(required)}.")

    geom = seg.apply(
        lambda r: LineString([(r["Inicio_Lon"], r["Inicio_Lat"]),
                              (r["Fin_Lon"], r["Fin_Lat"])]),
        axis=1,
    )
    gdf = gpd.GeoDataFrame(seg, geometry=geom, crs="EPSG:4326")
    return gdf.to_crs(target_crs), str(csv_path.name)


def load_accidents(target_crs):
    if not ACC_CSV.exists():
        fail("cyclist_accidents_enriched_v2.csv was not found.")
    acc = pd.read_csv(ACC_CSV)

    # In the pipeline, x = longitude and y = latitude (EPSG:4326).
    if {"x", "y"}.issubset(acc.columns):
        lon, lat = acc["x"], acc["y"]
    elif {"longitud", "latitud"}.issubset(acc.columns):
        lon, lat = acc["longitud"], acc["latitud"]
    elif {"Lon", "Lat"}.issubset(acc.columns):
        lon, lat = acc["Lon"], acc["Lat"]
    else:
        fail("No coordinate columns were found in the crash-record CSV.")

    gdf = gpd.GeoDataFrame(
        acc,
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    return gdf.to_crs(target_crs)


def add_scale_bar(ax, length_m=10000, location=(0.035, 0.055), segments=5):
    """Scale bar in map coordinates (metres)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_start = x0 + location[0] * (x1 - x0)
    y_pos = y0 + location[1] * (y1 - y0)
    step = length_m / segments

    for i in range(segments):
        color = "black" if i % 2 == 0 else "white"
        ax.plot(
            [x_start + i * step, x_start + (i + 1) * step],
            [y_pos, y_pos],
            color=color,
            linewidth=5,
            solid_capstyle="butt",
            zorder=20,
        )
    ax.plot(
        [x_start, x_start + length_m],
        [y_pos, y_pos],
        color="black",
        linewidth=0.8,
        zorder=21,
    )
    ax.text(
        x_start + length_m / 2,
        y_pos + 0.012 * (y1 - y0),
        f"{int(length_m / 1000)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=21,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2),
    )


def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.965, 0.945),
        xytext=(0.965, 0.865),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(facecolor="black", edgecolor="black", width=2.2, headwidth=8),
        zorder=30,
    )


def main():
    log("=" * 78)
    log("FIGURE 1 - STUDY-AREA MAP")
    log(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 78)

    if not DEM_CLIP.exists():
        fail("copernicus_gma_clip_v2.tif was not found. Run the v2.2 pipeline first.")

    with rasterio.open(DEM_CLIP) as src:
        elev = src.read(1).astype("float32")
        nodata = src.nodata
        crs = src.crs
        transform = src.transform
        if nodata is not None:
            elev[elev == nodata] = np.nan
        elev[elev <= 0] = np.nan
        res_x = abs(transform.a)
        res_y = abs(transform.e)
        x0 = transform.c
        y1 = transform.f
        x1 = x0 + elev.shape[1] * transform.a
        y0 = y1 + elev.shape[0] * transform.e

    valid = np.isfinite(elev)
    if valid.sum() == 0:
        fail("The clipped DEM contains no valid pixels.")

    log(f"DEM: {DEM_CLIP.name}")
    log(f"  CRS: {crs}")
    log(f"  Valid pixels: {valid.sum():,}")
    log(f"  Elevation: min={np.nanmin(elev):.1f} m, max={np.nanmax(elev):.1f} m")
    log(f"  Extent: {x0:.0f}-{x1:.0f} m E, {y0:.0f}-{y1:.0f} m N")

    segments, seg_source = load_segments(crs)
    accidents = load_accidents(crs)
    log(f"Segments: {len(segments):,} ({seg_source})")
    log(f"Crash records: {len(accidents):,} ({ACC_CSV.name})")

    # Clip vector layers to the raster extent for safety.
    from shapely.geometry import box
    raster_box = box(x0, y0, x1, y1)
    segments_plot = segments[segments.intersects(raster_box)].copy()
    accidents_plot = accidents[accidents.intersects(raster_box)].copy()
    log(f"Segments inside raster: {len(segments_plot):,}")
    log(f"Crash records inside raster: {len(accidents_plot):,}")

    # Shaded relief + elevation.
    elev_ma = np.ma.masked_invalid(elev)
    vmin, vmax = np.nanpercentile(elev, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("terrain")
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(
        elev_ma,
        cmap=cmap,
        norm=norm,
        blend_mode="soft",
        vert_exag=1.25,
        dx=res_x,
        dy=res_y,
    )

    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.set_facecolor("#f2f2f2")
    ax.imshow(rgb, extent=[x0, x1, y0, y1], origin="upper", interpolation="bilinear")

    segments_plot.plot(
        ax=ax,
        color="#00A6D6",
        linewidth=0.38,
        alpha=0.78,
        zorder=5,
    )
    ax.scatter(
        accidents_plot.geometry.x,
        accidents_plot.geometry.y,
        s=10,
        c="#D62828",
        edgecolors="white",
        linewidths=0.18,
        alpha=0.88,
        zorder=6,
    )

    # Main reference cities (WGS84 -> raster CRS).
    cities = gpd.GeoDataFrame(
        {
            "name": [
                "Guadalajara",
                "Zapopan",
                "Tlaquepaque",
                "Tonala",
                "La Primavera",
            ],
            "lon": [-103.3496, -103.4400, -103.3120, -103.2340, -103.5850],
            "lat": [20.6737, 20.7236, 20.6400, 20.6160, 20.6200],
        },
        geometry=gpd.points_from_xy(
            [-103.3496, -103.4400, -103.3120, -103.2340, -103.5850],
            [20.6737, 20.7236, 20.6400, 20.6160, 20.6200],
        ),
        crs="EPSG:4326",
    ).to_crs(crs)

    for _, row in cities.iterrows():
        xx, yy = row.geometry.x, row.geometry.y
        if x0 <= xx <= x1 and y0 <= yy <= y1:
            ax.plot(xx, yy, marker="o", markersize=2.6, color="black", zorder=8)
            ax.text(
                xx + 0.008 * (x1 - x0),
                yy + 0.006 * (y1 - y0),
                row["name"],
                fontsize=7.5,
                color="black",
                zorder=9,
                bbox=dict(facecolor="white", alpha=0.62, edgecolor="none", pad=1.0),
            )

    # Axes in km.
    km_formatter = FuncFormatter(lambda v, pos: f"{v / 1000:,.0f}")
    ax.xaxis.set_major_formatter(km_formatter)
    ax.yaxis.set_major_formatter(km_formatter)
    ax.set_xlabel("Easting (km, UTM zone 13N)", fontsize=9)
    ax.set_ylabel("Northing (km, UTM zone 13N)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")

    add_scale_bar(ax, length_m=10000)
    add_north_arrow(ax)

    # Colour bar (elevation).
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Elevation (m, GLO-30)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    legend_items = [
        Line2D([0], [0], color="#00A6D6", lw=1.4,
               label=f"Cycling network microsegments (n = {len(segments_plot):,})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D62828",
               markeredgecolor="white", markersize=5,
               label=f"Cyclist-involved crash records (n = {len(accidents_plot):,})"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.4",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    log("")
    log(f"Saved: {OUT_PNG.name} (300 dpi)")
    log(f"Saved: {OUT_PDF.name} (vector)")
    (BASE / OUT_LOG.name).write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
