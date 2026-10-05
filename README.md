# 🛰️ Spatial Apartheid: Quantifying Bengaluru's Educational Divide

**An open, reproducible geospatial pipeline auditing the Right to Education (RTE) walking-distance compliance in Bengaluru's formal and informal settlements.**

**Authors:** Kaveri Sharma, Vedika Chhabra, Jahnvi R, Manasa Ranganath, Komal Kumari, Dr. Divyashree N (PES University)

---

## 📌 Executive Summary
The Indian **Right to Education (RTE) Act** mandates a government primary school within a **1 km walking distance** of every child's habitation. Compliance is usually audited using outdated census frames and straight-line buffers, which ignore informal settlements and actual street networks. 

This project uses Sentinel-2 satellite imagery, OSM network data, the KSDB slum register, and WorldPop population estimates to audit true walking-distance compliance at scale.

**Key Findings:**
- **City-wide Compliance Gap:** Only **17.3%** of sampled locations and **20.1%** of slum residents lie within a 1 km walk of a government school.
- **The "Crow-Flies Fallacy":** Traditional straight-line buffers massively overstate compliance. About 48.2% of locations that appear compliant by a straight line are non-compliant by actual street network distance.
- **Formal vs. Informal:** Planned layouts are slightly farther from government schools on average than informal areas (2.32 km vs. 2.10 km), though the effect size is negligible. Both groups face long walking distances.
- **The Invisible Burden:** Non-notified slums (unrecognized by the state) are significantly farther from government schools (2.40 km) than notified slums (1.71 km). 
- **Concentrated Demand:** The ten most-loaded government schools serve as the nearest public option for 71.5% of all slum residents.

While 71–76% of locations lie within 1 km of *some* school (mostly private), free government options remain highly inaccessible by foot.

---

## 🛠️ Tech Stack & Methodology

Our pipeline ("Map → Detect → Measure → Weigh") moves beyond basic radius mapping:

1. **Map (Data Ingestion):**
   * Sentinel-2 L2A (10m multispectral imagery via Google Earth Engine).
   * OpenStreetMap (OSM) pedestrian graph (186,595 nodes, 499,950 edges) using `OSMnx`.
   * Curated OSM school features and Karnataka Slum Development Board (KSDB) slum polygons.
2. **Detect (AI Classification):**
   * Computed NDVI (Vegetation) and NDBI (Built-up) spectral indices.
   * Trained a **Random Forest Classifier** to segment planned grids vs. informal settlements, generating "virtual student" sample locations.
3. **Measure (Spatial Routing):**
   * NetworkX-powered **Labelled Multi-Source Dijkstra's Algorithm**.
   * Constructs a network Voronoi partition in seconds, returning both the exact walking distance and the assigned catchment school for every point on the graph.
4. **Weigh (Population Dynamics):**
   * Integrated WorldPop 2020 gridded estimates to compute population-weighted distances for all 509 KSDB slums and measure catchment loads ($\Lambda_s$) per school.

---

## 📂 Repository Structure

* `data/`: Contains geojson, shapefiles, KMLs, and other raw data (excluding large caching and graph files).
* `notebooks/`: Contains the core `Spatial_Apartheid_Phase1.ipynb` notebook detailing the end-to-end data pipeline.
* `scripts/`: Python scripts for analysis, processing, and visualization plotting.
* `results/`: Contains the generated catchment maps, vulnerability maps, charts, and statistical tables.
* `manuscript/`: Contains the LaTeX source code and PDF for the IEEE conference manuscript.

---

## 🚀 How to Run the Code

1. Clone the repository and install required Python packages (`osmnx`, `geopandas`, `networkx`, `earthengine-api`, etc.).
2. You will need a registered **Google Earth Engine** account to pull Sentinel-2 satellite data.
3. Open `notebooks/Spatial_Apartheid_Phase1.ipynb` to run the data pipeline. Follow the authentication prompts to link your Google Cloud Project.
4. You can also run the analysis directly using the scripts in the `scripts/` folder (e.g., `python scripts/paper_analysis.py`).
