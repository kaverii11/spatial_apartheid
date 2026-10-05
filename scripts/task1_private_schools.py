import osmnx as ox
import geopandas as gpd
import pandas as pd
import folium
import networkx as nx
from shapely.geometry import Point
import numpy as np

# Set CRS to UTM Zone 43N for Bengaluru
PROJECTED_CRS = "EPSG:32643"
WGS84_CRS = "EPSG:4326"

def get_schools_from_osm(place_name="Bengaluru, Karnataka, India"):
    """
    Fetches all schools from OSM for the given place.
    """
    tags = {'amenity': 'school'}
    schools = ox.features_from_place(place_name, tags)
    return schools

def filter_private_schools(schools_gdf):
    """
    Filters private schools based on common OSM tags and name heuristics.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = schools_gdf.copy()
    
    # 1. Check explicit tags if they exist
    private_cond = pd.Series(False, index=df.index)
    if 'fee' in df.columns:
        private_cond = private_cond | (df['fee'].str.lower() == 'yes')
    if 'operator:type' in df.columns:
        private_cond = private_cond | (df['operator:type'].str.lower() == 'private')
        
    # 2. Exclude government schools based on name heuristics
    # Lowercase names for easier matching
    names = df['name'].str.lower().fillna('')
    govt_keywords = ['government', 'govt', 'bbmp', 'ghps', 'glps', 'public school'] # Note: 'public school' in India often means private, but sometimes not. Let's stick to strict govt keywords.
    govt_keywords = ['government', 'govt', 'bbmp', 'ghps', 'glps']
    
    # Exclude if name contains govt keywords
    is_govt_by_name = names.apply(lambda x: any(kw in x for kw in govt_keywords))
    
    # Assume private if explicitly tagged or NOT government by name (conservative heuristic for private)
    # Refinement: In India, many private schools have "Public School" in name (e.g., Delhi Public School)
    # We will classify as private if it's explicitly private OR (has a name AND is not government)
    private_schools = df[private_cond | (~is_govt_by_name & (names != ''))].copy()
    
    # We only want points for routing. If they are polygons, get centroids
    private_schools['geometry'] = private_schools.geometry.centroid
    return private_schools

def filter_government_schools(schools_gdf):
    df = schools_gdf.copy()
    names = df['name'].str.lower().fillna('')
    govt_keywords = ['government', 'govt', 'bbmp', 'ghps', 'glps']
    is_govt_by_name = names.apply(lambda x: any(kw in x for kw in govt_keywords))
    govt_schools = df[is_govt_by_name].copy()
    govt_schools['geometry'] = govt_schools.geometry.centroid
    return govt_schools

def calculate_density(schools_gdf, settlements_gdf):
    """
    Calculates spatial density of schools within formal/informal settlements.
    Assumes settlements_gdf has a 'type' column ('Formal' or 'Informal')
    """
    # Ensure both are in the same projected CRS for area calculation
    schools_proj = schools_gdf.to_crs(PROJECTED_CRS)
    settlements_proj = settlements_gdf.to_crs(PROJECTED_CRS)
    
    # Perform spatial join to count schools in each settlement
    joined = gpd.sjoin(schools_proj, settlements_proj, how='inner', predicate='within')
    counts = joined.groupby('index_right').size()
    
    # Add count back to settlements
    settlements_proj['school_count'] = 0
    settlements_proj.loc[counts.index, 'school_count'] = counts.values
    
    # Calculate area in sq kilometers
    settlements_proj['area_sqkm'] = settlements_proj.geometry.area / 10**6
    
    # Calculate density
    settlements_proj['school_density_per_sqkm'] = settlements_proj['school_count'] / settlements_proj['area_sqkm']
    
    return settlements_proj

def calculate_network_distances(settlements_gdf, private_schools, govt_schools, place_name="Bengaluru, Karnataka, India"):
    """
    Calculates average network walking distance from Formal settlements to nearest private vs govt schools.
    """
    # Load pre-downloaded graph to save time
    graph_path = 'drive-download-20261003T103803Z-1-001/bengaluru_walk.graphml.xml'
    print(f"Loading local graph from {graph_path}...")
    G = ox.load_graphml(graph_path)
    
    # Ensure CRS matching
    settlements_wgs = settlements_gdf.to_crs(WGS84_CRS)
    private_wgs = private_schools.to_crs(WGS84_CRS)
    govt_wgs = govt_schools.to_crs(WGS84_CRS)
    
    # Get nodes for schools
    private_nodes = ox.distance.nearest_nodes(G, private_wgs.geometry.x, private_wgs.geometry.y)
    govt_nodes = ox.distance.nearest_nodes(G, govt_wgs.geometry.x, govt_wgs.geometry.y)
    
    results = []
    
    for idx, row in settlements_wgs.iterrows():
        # Get centroid of settlement
        centroid = row.geometry.centroid
        settlement_node = ox.distance.nearest_nodes(G, centroid.x, centroid.y)
        
        # Distance to nearest private school
        dist_private = min([nx.shortest_path_length(G, settlement_node, p_node, weight='length') for p_node in private_nodes] + [np.inf])
        
        # Distance to nearest govt school
        dist_govt = min([nx.shortest_path_length(G, settlement_node, g_node, weight='length') for g_node in govt_nodes] + [np.inf])
        
        results.append({
            'settlement_id': idx,
            'type': row.get('type', 'Unknown'),
            'dist_to_nearest_private_m': dist_private,
            'dist_to_nearest_govt_m': dist_govt
        })
        
    return pd.DataFrame(results)

def generate_map(settlements_gdf, private_schools):
    """
    Generates a Folium map showing clustering of private schools in formal "education deserts".
    """
    # Create base map centered on Bengaluru
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
    
    # Add settlements
    settlements_wgs = settlements_gdf.to_crs(WGS84_CRS)
    
    def style_function(feature):
        color = 'blue' if feature['properties'].get('type') == 'Formal' else 'red'
        return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.5}
        
    folium.GeoJson(
        settlements_wgs,
        name='Settlements',
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['type', 'school_density_per_sqkm'])
    ).add_to(m)
    
    # Add private schools as cluster or points
    private_wgs = private_schools.to_crs(WGS84_CRS)
    for idx, row in private_wgs.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color='green',
            fill=True,
            fill_color='green',
            popup=row.get('name', 'Private School')
        ).add_to(m)
        
    folium.LayerControl().add_to(m)
    return m

if __name__ == "__main__":
    place = "Bengaluru, Karnataka, India"
    print("Fetching schools from OSM...")
    all_schools = get_schools_from_osm(place)
    
    private_schools = filter_private_schools(all_schools)
    govt_schools = filter_government_schools(all_schools)
    print(f"Found {len(private_schools)} private schools and {len(govt_schools)} government schools.")
    
    # Using the real slums data found in the drive download folder!
    settlements_path = 'drive-download-20261003T103803Z-1-001/blr_slums.json'
    print(f"Loading real settlement polygons from {settlements_path}...")
    settlements_gdf = gpd.read_file(settlements_path)
    # Ensure CRS is WGS84 for OSM distance calculations
    settlements_gdf = settlements_gdf.to_crs(WGS84_CRS)
    
    print("Calculating density...")
    density_df = calculate_density(private_schools, settlements_gdf)
    
    print("Calculating network distances (this may take a while)...")
    distance_results = calculate_network_distances(settlements_gdf, private_schools, govt_schools, place)
    
    # Merge and output summary table
    summary_table = pd.merge(density_df.drop(columns=['geometry']), distance_results, left_index=True, right_on='settlement_id')
    summary_table.to_csv("task1_summary.csv", index=False)
    print("Summary table saved to task1_summary.csv")
    
    print("Generating map...")
    m = generate_map(density_df, private_schools)
    m.save("task1_map.html")
    print("Map saved to task1_map.html")
