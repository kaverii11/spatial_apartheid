import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import numpy as np

# Set CRS constants
PROJECTED_CRS = "EPSG:32643"  # UTM Zone 43N (Bengaluru)
WGS84_CRS = "EPSG:4326"

def calculate_population_per_polygon(settlements_gdf, pop_raster_path):
    """
    Calculates total population within each settlement polygon using WorldPop raster.
    """
    # Ensure settlements are in the same CRS as the raster for accurate zonal stats
    # Note: Usually WorldPop is in EPSG:4326 (WGS84). 
    # We will assume the raster is in WGS84, so we project polygons to WGS84 for the stats.
    settlements_wgs = settlements_gdf.to_crs(WGS84_CRS)
    
    # Calculate zonal stats (sum of pixel values = total population)
    stats = zonal_stats(
        settlements_wgs, 
        pop_raster_path, 
        stats="sum", 
        geojson_out=False,
        nodata=-99999 # Adjust based on WorldPop nodata value
    )
    
    # Extract sums
    pop_sums = [s['sum'] if s['sum'] is not None else 0 for s in stats]
    
    # Add to DataFrame
    settlements_gdf = settlements_gdf.copy()
    settlements_gdf['Population'] = pop_sums
    
    return settlements_gdf

def calculate_congestion_adjusted_accessibility(settlements_gdf, capacity_constant=300):
    """
    Calculates Congestion-Adjusted Distance.
    Assumes 'Network_Walking_Distance_to_School' and 'Population' columns exist.
    """
    df = settlements_gdf.copy()
    
    # Apply formula
    # To prevent division by zero or huge multipliers if pop is 0:
    df['Congestion_Multiplier'] = df['Population'].apply(lambda p: max(1, p) / capacity_constant)
    
    df['Congestion_Adjusted_Distance'] = df['Network_Walking_Distance_to_School'] * df['Congestion_Multiplier']
    
    return df

def plot_distance_vs_population(settlements_gdf):
    """
    Creates a scatter plot of Distance vs Population for Formal and Informal zones.
    """
    plt.figure(figsize=(10, 6))
    
    formal = settlements_gdf[settlements_gdf['type'] == 'Formal']
    informal = settlements_gdf[settlements_gdf['type'] == 'Informal']
    
    plt.scatter(formal['Population'], formal['Network_Walking_Distance_to_School'], 
                alpha=0.6, label='Formal Settlements', color='blue')
    plt.scatter(informal['Population'], informal['Network_Walking_Distance_to_School'], 
                alpha=0.6, label='Informal Settlements', color='red')
    
    plt.title('Walking Distance vs. Population (Settlement Level)', fontsize=14)
    plt.xlabel('Settlement Population', fontsize=12)
    plt.ylabel('Network Walking Distance to Nearest School (m)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('task3_distance_vs_population.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    # --- Instructions to fetch WorldPop ---
    print("To fetch WorldPop data for Bengaluru (India):")
    print("1. Visit: https://hub.worldpop.org/geodata/summary?id=29472 (or search India 100m population)")
    print("2. Download the GeoTIFF (e.g., ind_ppp_2020_1km.tif or 100m version).")
    print("3. Place it in your working directory.\n")
    
    # Placeholder for raster path
    raster_path = "worldpop_bengaluru_simulated.tif"
    
    # --- SIMULATE RASTER FOR DEMONSTRATION IF IT DOESN'T EXIST ---
    import os
    if not os.path.exists(raster_path):
        print(f"Creating a dummy raster {raster_path} for demonstration...")
        import rasterio
        from rasterio.transform import from_origin
        
        # Create a dummy raster in WGS84 around Bengaluru
        transform = from_origin(77.5, 13.0, 0.001, 0.001)
        data = np.random.rand(100, 100) * 50 # 0 to 50 people per pixel
        
        with rasterio.open(
            raster_path, 'w', driver='GTiff',
            height=data.shape[0], width=data.shape[1],
            count=1, dtype=data.dtype,
            crs='+proj=latlong', transform=transform
        ) as dst:
            dst.write(data, 1)
    
    # --- SIMULATE SETTLEMENT POLYGONS ---
    from shapely.geometry import Polygon
    simulated_data = {
        'type': ['Formal', 'Informal', 'Formal', 'Informal'],
        'Network_Walking_Distance_to_School': [400, 1200, 600, 1500],
        'geometry': [
            Polygon([(77.51, 12.91), (77.52, 12.91), (77.52, 12.92), (77.51, 12.92)]),
            Polygon([(77.53, 12.93), (77.54, 12.93), (77.54, 12.94), (77.53, 12.94)]),
            Polygon([(77.55, 12.95), (77.56, 12.95), (77.56, 12.96), (77.55, 12.96)]),
            Polygon([(77.57, 12.97), (77.58, 12.97), (77.58, 12.98), (77.57, 12.98)])
        ]
    }
    settlements_gdf = gpd.GeoDataFrame(simulated_data, crs=WGS84_CRS)
    # Project to UTM 43N for distance metric processing internally if needed, but WorldPop is usually WGS84.
    
    print("Calculating population per polygon...")
    settlements_with_pop = calculate_population_per_polygon(settlements_gdf, raster_path)
    
    print("Calculating congestion-adjusted accessibility...")
    final_gdf = calculate_congestion_adjusted_accessibility(settlements_with_pop, capacity_constant=300)
    
    print("\n--- Processed Data Sample ---")
    print(final_gdf[['type', 'Population', 'Network_Walking_Distance_to_School', 'Congestion_Adjusted_Distance']])
    
    # Output to GeoJSON
    output_geojson = "task3_congestion_scores.geojson"
    final_gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"\nSaved GeoJSON to {output_geojson}")
    
    print("Generating scatter plot...")
    plot_distance_vs_population(final_gdf)
    print("Saved plot to task3_distance_vs_population.png")
