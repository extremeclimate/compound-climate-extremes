import os
import time
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import numpy as np
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SHP_CITIES = os.path.join(DATA_DIR, "shapefiles", "cleaned_cities.shp")
SHP_WORLD_CORRECT_CHINA = os.path.join(DATA_DIR, "shapefiles", "world_boundaries.shp")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = [str(y) for y in range(1990, 2025)]

CSV_PATHS = {
    "Dry_Wind_Heat": os.path.join(DATA_DIR, "triple_events", "dry_wind_heat.csv"),
    "Dry_Wind_Cold": os.path.join(DATA_DIR, "triple_events", "dry_wind_cold.csv"),
    "Dry_Heat_WindDry": os.path.join(DATA_DIR, "triple_events", "dry_heat_wind_drought.csv"),
    "Rain_Cold_WindDry": os.path.join(DATA_DIR, "triple_events", "rain_cold_wind_drought.csv"),
    "Rain_Wind_Cold": os.path.join(DATA_DIR, "triple_events", "rain_wind_cold.csv"),
    "Dry_Cold_WindDry": os.path.join(DATA_DIR, "triple_events", "dry_cold_wind_drought.csv"),
    "Rain_Heat_WindDry": os.path.join(DATA_DIR, "triple_events", "rain_heat_wind_drought.csv"),
    "Rain_Wind_Heat": os.path.join(DATA_DIR, "triple_events", "rain_wind_heat.csv"),
    "Dry_Rain_Cold": os.path.join(DATA_DIR, "triple_events", "dry_rain_cold.csv"),
    "Dry_Rain_Heat": os.path.join(DATA_DIR, "triple_events", "dry_rain_heat.csv"),
    "Dry_Rain_WindDry": os.path.join(DATA_DIR, "triple_events", "dry_rain_wind_drought.csv"),
    "Dry_Rain_Wind": os.path.join(DATA_DIR, "triple_events", "dry_rain_wind.csv")
}

EVENTS_CONFIG = [
    {"key": "Dry_Wind_Heat", "title": "a DHW", "cmap": "YlOrRd", "vmax": 2.5},
    {"key": "Dry_Wind_Cold", "title": "b DCW", "cmap": "YlGnBu", "vmax": 1.2},
    {"key": "Dry_Heat_WindDry", "title": "c DHWd", "cmap": "OrRd", "vmax": 1.0},
    {"key": "Rain_Cold_WindDry", "title": "d RCWd", "cmap": "PuBu", "vmax": 0.8},
    {"key": "Rain_Wind_Cold", "title": "e RCW", "cmap": "BuPu", "vmax": 0.6},
    {"key": "Dry_Cold_WindDry", "title": "f DCWd", "cmap": "Purples", "vmax": 0.4},
    {"key": "Rain_Heat_WindDry", "title": "g RHWd", "cmap": "RdPu", "vmax": 0.4},
    {"key": "Rain_Wind_Heat", "title": "h RHW", "cmap": "Oranges", "vmax": 0.4},
    {"key": "Dry_Rain_Cold", "title": "i DRC", "cmap": "GnBu", "vmax": 0.4},
    {"key": "Dry_Rain_Heat", "title": "j DRH", "cmap": "YlOrBr", "vmax": 0.4},
    {"key": "Dry_Rain_WindDry", "title": "k DRWd", "cmap": "PuRd", "vmax": 0.4},
    {"key": "Dry_Rain_Wind", "title": "l DRW", "cmap": "BuGn", "vmax": 0.4}
]

def load_and_aggregate_data():
    print("Loading foundational geographic data...")
    gdf_cities = gpd.read_file(SHP_CITIES)
    gdf_cities['lat_true'] = gdf_cities.geometry.centroid.y
    gdf_cities = gdf_cities.to_crs('+proj=robin')

    gdf_world_correct_china = gpd.read_file(SHP_WORLD_CORRECT_CHINA)
    if gdf_world_correct_china.crs is None:
        gdf_world_correct_china = gdf_world_correct_china.set_crs(epsg=4326)
    gdf_world_correct_china = gdf_world_correct_china.to_crs('+proj=robin')

    print("Simplifying polygons to turbocharge rendering...")
    gdf_cities['geometry'] = gdf_cities['geometry'].simplify(tolerance=10000, preserve_topology=True)

    global_trends = {}
    print("Loading and calculating 35-year CUMULATIVE sums & Trends...")
    for config in EVENTS_CONFIG:
        key = config["key"]
        path = CSV_PATHS[key]
        try:
            df = pd.read_csv(path, index_col=0).rename_axis('NAME_2').reset_index()
            df.columns = df.columns.astype(str)
            valid_cols = [y for y in YEARS if y in df.columns]
            df[f"{key}_Cumulative"] = df[valid_cols].sum(axis=1)
            global_trends[key] = df[valid_cols].sum(axis=0)
            gdf_cities = gdf_cities.merge(df[['NAME_2', f"{key}_Cumulative"]], on='NAME_2', how='left')
        except Exception as e:
            print(f"Error processing {key}: {e}")
            gdf_cities[f"{key}_Cumulative"] = np.nan
            global_trends[key] = None

    return gdf_cities, gdf_world_correct_china, global_trends

def plot_12_events_matrix(gdf_plot, gdf_world_correct_china, global_trends):
    print("Starting rendering engine for 12 Triple Compound Events...")
    fig = plt.figure(figsize=(28, 19), facecolor='white')

    w_map, w_lat_gap, w_lat, w_group_gap = 1.0, 0.01, 0.12, 0.35
    gs = gridspec.GridSpec(4, 11, height_ratios=[1] * 4,
                           width_ratios=[w_map, w_lat_gap, w_lat, w_group_gap,
                                         w_map, w_lat_gap, w_lat, w_group_gap,
                                         w_map, w_lat_gap, w_lat],
                           hspace=0.78, wspace=0.0)

    lat_bins = np.arange(-60, 86, 2)
    gdf_plot['lat_bin'] = pd.cut(gdf_plot['lat_true'], bins=lat_bins, labels=lat_bins[:-1])

    with tqdm(total=12, desc="Rendering 12 Maps") as pbar:
        for i, config in enumerate(EVENTS_CONFIG):
            row_idx, col_block = i // 3, i % 3
            map_col_idx, lat_col_idx = col_block * 4, col_block * 4 + 2
            col_target = f"{config['key']}_Cumulative"

            ax = fig.add_subplot(gs[row_idx, map_col_idx], projection=ccrs.Robinson())
            ax.set_global()
            ax.spines['geo'].set_visible(True)
            ax.spines['geo'].set_edgecolor('black')
            ax.spines['geo'].set_linewidth(0.5)
            ax.patch.set_facecolor('#FFFFFF')

            gdf_world_correct_china.plot(ax=ax, facecolor='#F5F5F5', edgecolor='none', zorder=0)

            gdf_valid = gdf_plot.dropna(subset=[col_target])
            if not gdf_valid.empty:
                gdf_valid.plot(column=col_target, ax=ax, cmap=config['cmap'],
                               linewidth=0.0, edgecolor='none',
                               vmin=0, vmax=config['vmax'], zorder=1)

            gdf_world_correct_china.plot(ax=ax, facecolor='none', edgecolor='#333333', linewidth=0.75, zorder=6)

            title_text = config['title'] + (' (Total Events)' if i == 0 else '')
            ax.text(0.0, 1.34, title_text, transform=ax.transAxes,
                    fontsize=24, fontweight='normal', color='black', ha='left')
            cax = ax.inset_axes([0.0, 1.15, 0.45, 0.05])
            sm = ScalarMappable(cmap=config['cmap'], norm=Normalize(vmin=0, vmax=config['vmax']))
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=16, length=3, pad=3, colors='black')
            for tick_label in cbar.ax.get_xticklabels():
                tick_label.set_fontfamily('Arial')
                tick_label.set_fontweight('normal')
                tick_label.set_color('black')

            ax_lat = fig.add_subplot(gs[row_idx, lat_col_idx])
            if not gdf_valid.empty:
                z_max = gdf_plot.groupby('lat_bin', observed=False)[col_target].max().fillna(0)
                zx_s = z_max.rolling(5, center=True, min_periods=1).mean()
                y_p = ccrs.Robinson().transform_points(ccrs.PlateCarree(), np.zeros_like(zx_s),
                                                       np.array(zx_s.index) + 1)[:, 1]
                ax_lat.plot(zx_s.values, y_p, color='black', linewidth=1.8,
                            linestyle='-', zorder=2)

            ax_lat.set_ylim(ax.get_ylim())
            for spine in ax_lat.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.5)
            ax_lat.yaxis.tick_right()
            ax_lat.yaxis.set_label_position("right")
            ax_lat.set_xlim(left=0)
            ax_lat.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            ax_lat.tick_params(axis='both', labelsize=16, colors='black')

            t_lats = np.array([-60, -30, 0, 30, 60])
            t_y_p = ccrs.Robinson().transform_points(ccrs.PlateCarree(), np.zeros_like(t_lats), t_lats)[:, 1]
            ax_lat.set_yticks(t_y_p)
            ax_lat.set_yticklabels([
                f"{abs(l)}°S" if l < 0 else f"{l}°N" if l > 0 else "0°" for l in t_lats
            ])
            for tick_label in ax_lat.get_xticklabels() + ax_lat.get_yticklabels():
                tick_label.set_fontfamily('Arial')
                tick_label.set_fontweight('normal')
                tick_label.set_color('black')

            pbar.update(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_tif = os.path.join(OUTPUT_DIR, f"Figure_Nature_Style_Matrix_correct_china_{timestamp}.tif")
    output_svg = os.path.join(OUTPUT_DIR, f"Figure_Nature_Style_Matrix_correct_china_{timestamp}.svg")

    total_save_start = time.time()

    print("Saving TIFF (300 DPI, LZW compressed)...", flush=True)
    tif_start = time.time()
    plt.savefig(output_tif, dpi=300, bbox_inches='tight', facecolor='white',
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"TIFF saved successfully! Time used: {(time.time() - tif_start) / 60:.2f} min", flush=True)
    print(f"TIFF path: {output_tif}", flush=True)

    print("Saving SVG...", flush=True)
    svg_start = time.time()
    plt.savefig(output_svg, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"SVG saved successfully! Time used: {(time.time() - svg_start) / 60:.2f} min", flush=True)
    print(f"SVG path: {output_svg}", flush=True)

    plt.close(fig)
    print(f"Success! High-quality matrix saved to:\n{OUTPUT_DIR}")
    print(f"Timestamp: {timestamp}")
    print(f"Total saving time: {(time.time() - total_save_start) / 60:.2f} min", flush=True)

if __name__ == "__main__":
    g_plot, g_world_correct_china, g_trends = load_and_aggregate_data()
    plot_12_events_matrix(g_plot, g_world_correct_china, g_trends)
