"""
Figure 2 - Pipeline workflow.

Arrow routing separates vector inputs (IIEG Jalisco crash records and the
GDL en Bici network) from raster DEM inputs (Copernicus GLO-30 and NASADEM).
Crash records do not enter DEM reprojection; they feed vector preparation, the
dynamic clipping extent, and direct sampling.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parent
OUT_PNG = BASE / "figura2_workflow.png"
OUT_PDF = BASE / "figura2_workflow.pdf"

C_INPUT = "#EAF3FB"
C_PROC = "#F2F2F2"
C_VALID = "#FFF3E0"
C_OUT = "#E8F5E9"
C_EDGE = "#4A4A4A"

fig, ax = plt.subplots(figsize=(12.5, 10.0))
ax.set_xlim(0, 12.5)
ax.set_ylim(-0.35, 10.1)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=9.2, weight="normal"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        linewidth=1.1, edgecolor=C_EDGE, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="black", weight=weight, linespacing=1.25)
    return (x, y, w, h)


def arrow(b1, b2, start="bottom", end="top", rad=0.0):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {
        "bottom": (x1 + w1 / 2, y1),
        "top": (x1 + w1 / 2, y1 + h1),
        "left": (x1, y1 + h1 / 2),
        "right": (x1 + w1, y1 + h1 / 2),
    }
    pte = {
        "bottom": (x2 + w2 / 2, y2),
        "top": (x2 + w2 / 2, y2 + h2),
        "left": (x2, y2 + h2 / 2),
        "right": (x2 + w2, y2 + h2 / 2),
    }
    a = FancyArrowPatch(
        pts[start], pte[end],
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.0, color=C_EDGE,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3, shrinkB=3,
    )
    ax.add_patch(a)


# Row 1: inputs
b_iieg = box(0.45, 8.75, 2.55, 1.05,
             "IIEG Jalisco\n774 crash records\n(2015-2024, CC BY 4.0)",
             C_INPUT, fs=9.0)
b_gdl = box(3.35, 8.75, 2.55, 1.05,
            "GDL en Bici\n13,380 microsegments\n(~20 m, CC BY 4.0)",
            C_INPUT, fs=9.0)
b_cop = box(6.55, 8.75, 2.65, 1.05,
            "Copernicus GLO-30\nX-band DEM, ~30 m\n(EGM2008)",
            C_INPUT, fs=9.0)
b_nasa = box(9.75, 8.75, 2.30, 1.05,
             "NASADEM\nC-band DEM, ~30 m\n(EGM96)",
             C_INPUT, fs=9.0)

# Row 2: vector lane (left) and raster lane (right)
b_vec = box(0.60, 6.85, 4.80, 1.00,
            "Vector preparation: EPSG:4326 to UTM 13N\nsegments as LineStrings; crash records as points",
            C_PROC, fs=9.2, weight="bold")
b_dem = box(7.00, 6.85, 5.00, 1.00,
            "DEM reprojection to UTM 13N, 30 m, bilinear\nexplicit destination nodata (-32,768)",
            C_PROC, fs=9.2, weight="bold")

# Raster lane: dynamic clipping and features
b_clip = box(6.70, 5.15, 5.60, 0.95,
             "Dynamic clip: joint extent of segments + crashes\n+ 2,000 m margin, snapped to 30 m grid",
             C_PROC, fs=9.2, weight="bold")
b_feat = box(7.10, 3.75, 4.60, 0.95,
             "Feature raster (GLO-30)\nslope (Horn, % rise)\nTRI, roughness, aspect",
             C_PROC, fs=8.8)

# Vector lane: two tiers
b_t1 = box(0.70, 3.75, 4.90, 0.95,
           "Tier 1: zonal statistics\n13,380 segments\n(rasterstats)",
           C_PROC, fs=9.2, weight="bold")
b_t2 = box(0.70, 2.25, 4.90, 0.95,
           "Tier 2: direct sampling\n774 crash locations\n(point + 30 m buffer)",
           C_PROC, fs=9.2, weight="bold")

# Validation
b_val = box(2.30, 0.75, 8.00, 0.85,
            "Technical validation: cross-DEM (r = 0.9995), logical checks,\n"
            "reference-zone evaluation, automated audit (95 checks, 0 errors)",
            C_VALID, fs=9.0, weight="bold")

# Outputs
b_zen = box(1.55, -0.25, 4.35, 0.72,
            "Zenodo: 3 CSV + metadata (SHA-256)\n+ codebook | CC BY 4.0",
            C_OUT, fs=8.8, weight="bold")
b_git = box(6.75, -0.25, 4.35, 0.72,
            "GitHub: pipeline, audit and diagnostic\nscripts | Apache 2.0",
            C_OUT, fs=8.8, weight="bold")

# Input arrows
arrow(b_iieg, b_vec, rad=-0.08)
arrow(b_gdl, b_vec, rad=0.08)
arrow(b_cop, b_dem, rad=-0.05)
arrow(b_nasa, b_dem, rad=0.05)

# Raster lane
arrow(b_dem, b_clip)
arrow(b_clip, b_feat)

# The dynamic bbox also uses the vector envelope
arrow(b_vec, b_clip, start="right", end="left", rad=0.04)

# Vector lane to the tiers (without crossing the raster lane)
arrow(b_vec, b_t1, start="bottom", end="top", rad=0.0)
arrow(b_vec, b_t2, start="left", end="left", rad=0.12)

# Topographic features to both tiers
arrow(b_feat, b_t1, start="left", end="right", rad=0.0)
arrow(b_feat, b_t2, start="left", end="right", rad=0.10)

# Validation
arrow(b_t1, b_val, start="bottom", end="top", rad=-0.06)
arrow(b_t2, b_val, start="bottom", end="top", rad=0.08)
arrow(b_clip, b_val, start="right", end="right", rad=-0.22)

# Outputs
arrow(b_val, b_zen, rad=-0.05)
arrow(b_val, b_git, rad=0.05)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)
print(f"Generados: {OUT_PNG.name}, {OUT_PDF.name}")
