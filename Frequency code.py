import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm
import numpy as np
from datetime import datetime
from matplotlib.ticker import MaxNLocator


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


SHP_WORLD_CORRECT_CHINA = os.path.join(DATA_DIR, "shapefiles", "world_map_correct_china.shp")

CSV_HEAT = os.path.join(DATA_DIR, "events", "temperature_extreme_heat_events_1990_2024.csv")
CSV_COLD = os.path.join(DATA_DIR, "events", "temperature_extreme_cold_events_1990_2024.csv")
CSV_RAIN = os.path.join(DATA_DIR, "events", "precipitation_heavy_rain_events_1990_2024.csv")
CSV_DROUGHT = os.path.join(DATA_DIR, "events", "drought_event_frequency.csv")
CSV_WIND = os.path.join(DATA_DIR, "events", "strong_wind_events_1990_2024.csv")
CSV_WIND_DROUGHT = os.path.join(DATA_DIR, "events", "wind_drought_events_1990_2024.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERIODS = {
    '1990-1999': [str(y) for y in range(1990, 2000)],
    '2000-2009': [str(y) for y in range(2000, 2010)],
    '2010-2019': [str(y) for y in range(2010, 2020)],
    '2020-2024': [str(y) for y in range(2020, 2025)]
}


def load_and_aggregate_data():
    print("Loading geographic data...")
    gdf_cities = gpd.read_file(SHP_CITIES)
    gdf_cities['lat_true'] = gdf_cities.geometry.centroid.y
    gdf_cities = gdf_cities.to_crs('+proj=robin')


    gdf_world_correct_china = gpd.read_file(SHP_WORLD_CORRECT_CHINA)
    if gdf_world_correct_china.crs is None:
        print("Warning: world-correct-China shp has no CRS; set to EPSG:4326 by default.")
        gdf_world_correct_china = gdf_world_correct_china.set_crs(epsg=4326)
    gdf_world_correct_china = gdf_world_correct_china.to_crs('+proj=robin')

    print("Simplifying polygons to turbocharge rendering...")
    gdf_cities['geometry'] = gdf_cities['geometry'].simplify(tolerance=10000, preserve_topology=True)

    print("Loading 6 panel datasets...")
    df_heat = pd.read_csv(CSV_HEAT)
    df_cold = pd.read_csv(CSV_COLD)
    df_rain = pd.read_csv(CSV_RAIN)
    df_drought = pd.read_csv(CSV_DROUGHT)
    df_wind = pd.read_csv(CSV_WIND, index_col=0).rename_axis('NAME_2').reset_index()
    df_wind_drought = pd.read_csv(CSV_WIND_DROUGHT, index_col=0).rename_axis('NAME_2').reset_index()

    dfs = [df_heat, df_cold, df_rain, df_drought, df_wind, df_wind_drought]
    for df in dfs:
        df.columns = df.columns.astype(str)

    def calc_period_means(df, prefix):
        res = df[['NAME_2']].copy()
        for p_name, years in PERIODS.items():
            valid_cols = [y for y in years if y in df.columns]
            res[f"{prefix}_{p_name}"] = df[valid_cols].mean(axis=1)
        return res

    print("Aggregating temporal data...")
    agg_heat = calc_period_means(df_heat, 'HEAT')
    agg_cold = calc_period_means(df_cold, 'COLD')
    agg_rain = calc_period_means(df_rain, 'RAIN')
    agg_drought = calc_period_means(df_drought, 'DROUGHT')
    agg_wind = calc_period_means(df_wind, 'WIND')
    agg_wind_drought = calc_period_means(df_wind_drought, 'WIND_DROUGHT')

    print("Merging spatial and aggregated data...")
    gdf_plot = gdf_cities.merge(agg_heat, on='NAME_2', how='left') \
        .merge(agg_cold, on='NAME_2', how='left') \
        .merge(agg_rain, on='NAME_2', how='left') \
        .merge(agg_drought, on='NAME_2', how='left') \
        .merge(agg_wind, on='NAME_2', how='left') \
        .merge(agg_wind_drought, on='NAME_2', how='left')

    return gdf_plot, gdf_world_correct_china


def plot_6x4_matrix(gdf_plot, gdf_world_correct_china):
    print("Starting rendering engine...")
    fig = plt.figure(figsize=(24, 28), facecolor='white')


    w_spacer = 0.05
    w_lat = 0.18

    gs = gridspec.GridSpec(7, 6, height_ratios=[0.5, 3, 3, 3, 3, 3, 3],
                           width_ratios=[1, 1, 1, 1, w_spacer, w_lat],
                           hspace=0.68, wspace=0.0)

    decades = list(PERIODS.keys())

    rows_config = [
        {'prefix': 'HEAT', 'cmap': 'YlOrRd', 'title': 'a) Heat Wave (Events/Year)', 'vmax': 4.0},
        {'prefix': 'COLD', 'cmap': 'Blues', 'title': 'b) Cold Wave (Events/Year)', 'vmax': 4.0},
        {'prefix': 'RAIN', 'cmap': 'GnBu', 'title': 'c) Heavy Rainfall (Events/Year)', 'vmax': 3.0},
        {'prefix': 'DROUGHT', 'cmap': 'OrRd', 'title': 'd) Drought (Events/Year)', 'vmax': 4.0},
        {'prefix': 'WIND', 'cmap': 'YlGn', 'title': 'e) Strong Wind (Events/Year)', 'vmax': 3.0},
        {'prefix': 'WIND_DROUGHT', 'cmap': 'BuPu', 'title': 'f) Wind Drought / Stagnation (Events/Year)', 'vmax': 2.0}
    ]


    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')


    total_w = 4.0 + w_spacer + w_lat
    centers = [(i + 0.5) / total_w for i in range(4)]

    ax_timeline.set_xlim(0, 1)
    ax_timeline.plot([centers[0], centers[-1]], [0.5, 0.5], color='#888888', lw=2)

    for i, dec in enumerate(decades):
        ax_timeline.plot([centers[i], centers[i]], [0.4, 0.6], color='black', lw=2)
        ax_timeline.text(centers[i], 0.7, dec, ha='center', va='bottom', fontsize=32,
                         fontweight='normal', color='black')

    total_maps = len(rows_config) * len(decades)
    map_axes_collection = []

    lat_bins = np.arange(-60, 86, 2)
    gdf_plot['lat_bin'] = pd.cut(gdf_plot['lat_true'], bins=lat_bins, labels=lat_bins[:-1])

    with tqdm(total=total_maps, desc="Rendering 24 Maps",
              bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for row_idx, config in enumerate(rows_config):
            col_targets_for_row = []
            map_ax_for_ylim = None

            for col_idx, dec in enumerate(decades):
                col_target = f"{config['prefix']}_{dec}"
                col_targets_for_row.append(col_target)

                ax = fig.add_subplot(gs[row_idx + 1, col_idx], projection=ccrs.Robinson())
                ax.set_global()

                if map_ax_for_ylim is None:
                    map_ax_for_ylim = ax

                ax.spines['geo'].set_visible(True)
                ax.spines['geo'].set_edgecolor('black')
                ax.spines['geo'].set_linewidth(0.5)
                ax.patch.set_facecolor('#FFFFFF')

                gdf_valid = gdf_plot.dropna(subset=[col_target])
                gdf_valid.plot(column=col_target, ax=ax, cmap=config['cmap'],
                               linewidth=0.0, edgecolor='none',
                               vmin=0, vmax=config['vmax'], zorder=1)


                gdf_world_correct_china.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='#333333',
                    linewidth=0.60,
                    zorder=5
                )

                map_axes_collection.append(ax)

                if col_idx == 0:
                    ax.text(0.0, 1.36, config['title'], transform=ax.transAxes, fontsize=30,
                            fontweight='normal', color='black', ha='left')

                    cax = ax.inset_axes([0.0, 1.14, 0.45, 0.045])
                    sm = ScalarMappable(cmap=config['cmap'], norm=Normalize(vmin=0, vmax=config['vmax']))
                    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=18, length=3, pad=3, colors='black')
                    for tick_label in cbar.ax.get_xticklabels():
                        tick_label.set_fontfamily('Arial')
                        tick_label.set_fontweight('normal')
                        tick_label.set_color('black')

                pbar.update(1)


            ax_lat = fig.add_subplot(gs[row_idx + 1, 5])

            gdf_plot['row_mean'] = gdf_plot[col_targets_for_row].mean(axis=1)
            gdf_plot['row_max'] = gdf_plot[col_targets_for_row].max(axis=1)

            zonal_mean = gdf_plot.groupby('lat_bin', observed=False)['row_mean'].mean().fillna(0)
            zonal_max = gdf_plot.groupby('lat_bin', observed=False)['row_max'].max().fillna(0)

            zonal_mean_smooth = zonal_mean.rolling(window=3, center=True, min_periods=1).mean()
            zonal_max_smooth = zonal_max.rolling(window=3, center=True, min_periods=1).mean()
            lats_center = np.array(zonal_mean_smooth.index) + 1

            y_proj = ccrs.Robinson().transform_points(ccrs.PlateCarree(), np.zeros_like(lats_center), lats_center)[:, 1]

            ax_lat.plot(zonal_max_smooth.values, y_proj, color='#999999', linewidth=1.2, linestyle='--', zorder=2,
                        label='Max')
            ax_lat.plot(zonal_mean_smooth.values, y_proj, color='#444444', linewidth=1.8, zorder=3, label='Mean')

            ax_lat.set_ylim(map_ax_for_ylim.get_ylim())


            for spine in ax_lat.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.5)

            ax_lat.yaxis.tick_right()
            ax_lat.yaxis.set_label_position("right")

            ax_lat.set_xlim(left=0)

            ax_lat.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_lat.tick_params(axis='x', labelsize=18, colors='black')
            ax_lat.tick_params(axis='y', labelsize=18, colors='black')
            ax_lat.set_ylabel('')

            tick_lats = np.array([-60, -30, 0, 30, 60])
            tick_y_proj = ccrs.Robinson().transform_points(ccrs.PlateCarree(), np.zeros_like(tick_lats), tick_lats)[:,
                          1]
            ax_lat.set_yticks(tick_y_proj)
            ax_lat.set_yticklabels([f"{abs(lat)}°{'S' if lat < 0 else 'N' if lat > 0 else ''}" for lat in tick_lats])
            for tick_label in ax_lat.get_xticklabels() + ax_lat.get_yticklabels():
                tick_label.set_fontfamily('Arial')
                tick_label.set_fontweight('normal')
                tick_label.set_color('black')


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Saving Version: Without City Boundaries only...")
    output_tif_no_bounds = os.path.join(
        OUTPUT_DIR, f"Figure_Base_Single_6x4_NoBounds_correct_china_{timestamp}.tif"
    )
    output_svg_no_bounds = os.path.join(
        OUTPUT_DIR, f"Figure_Base_Single_6x4_NoBounds_correct_china_{timestamp}.svg"
    )
    plt.savefig(output_tif_no_bounds, dpi=300, bbox_inches='tight', facecolor='white',
                pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(output_svg_no_bounds, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Success! Without City Boundaries version saved as TIFF (LZW, 300 DPI) and SVG to:\n{OUTPUT_DIR}")
    print(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    gdf_plot, gdf_world_correct_china = load_and_aggregate_data()
    plot_6x4_matrix(gdf_plot, gdf_world_correct_china)
