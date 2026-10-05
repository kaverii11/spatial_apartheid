# 🛰️ Spatial Apartheid: Quantifying Bengaluru's Educational Divide

**An AI and Geospatial Analytics project analyzing the true walking distance to public primary schools across formal and informal urban settlements.**

---

## 📌 Executive Summary
The Indian **Right to Education (RTE) Act** mandates that every child must have a primary school within a **1 km walking distance**. This project utilizes satellite imagery, machine learning, and massive-scale graph routing to audit Bengaluru's compliance with this mandate. 

**Our Initial Hypothesis:** Unmapped, informal settlements (slums) are "Education Deserts" cut off from public infrastructure.
**The Data-Driven Reality:** The exact opposite.

Through spatial analysis of over 300,000 streets, we discovered a profound **two-tiered segregation** in the city's education grid. Informal settlements are highly clustered around government schools (0.41 km average walk). In contrast, planned, wealthy layouts have completely abandoned the public grid (1.24 km average walk), relying entirely on a shadow network of private institutions. 

**The real Spatial Apartheid is not about physical distance; it is the systemic geographic isolation of public vs. private education.**

---

## 🛠️ Tech Stack & Methodology

This project moves beyond "As the Crow Flies" radius mapping by calculating true, street-level pedestrian routing. 

1. **Satellite Feature Engineering (Google Earth Engine):**
   * Pulled cloud-free Sentinel-2 Surface Reflectance Data.
   * Engineered spectral indices: **NDVI** (Vegetation) and **NDBI** (Built-up/Concrete).
2. **AI Image Classification (Machine Learning):**
   * Trained a **Random Forest Classifier (10 Trees)** directly on the satellite bands to segment the city into "Planned Grids" vs. "Informal Settlements."
   * Deployed 2,000 "virtual students" (coordinate points) into the classified zones for sampling.
3. **Open Data Ingestion (OpenStreetMap):**
   * Extracted the entire walkable street network of Bengaluru (300,000+ edges) using `OSMnx`.
   * Filtered OpenStreetMap POI data strictly for **Government Primary Schools** to match the state mandate.
4. **Massive-Scale Spatial Routing (NetworkX):**
   * Utilized **Multi-Source Dijkstra's Algorithm** to simultaneously flood the street graph from all 482 public schools, calculating the exact minimum walking distance to every sampled student coordinate in seconds.

---

## 📊 The Findings: The Public-Private Divide

<img width="2930" height="1545" alt="spatial_divide_chart" src="https://github.com/user-attachments/assets/d6feb351-799b-4329-a871-f9fceef6f71e" /> 

**Results:**
* **Informal Settlements (Unmapped Zones):** `0.41 km` average walk. (Highly Compliant)
* **Planned Layouts (Formal Grids):** `1.24 km` average walk. (Non-Compliant)

**Conclusion:** The data proves that the government does not build public primary schools in wealthy, planned layouts because the residents there utilize private schools. As a result, public infrastructure is historically and systematically isolated within or adjacent to low-income and informal settlements. 

---

## 📂 Repository Structure

* `Spatial_Apartheid_Phase1.ipynb`: The core Google Colab / Jupyter Notebook containing the entire end-to-end data pipeline, from Earth Engine authentication to NetworkX routing.
* `spatial_divide_chart.png`: The final Matplotlib/Seaborn visualization of the walking distance disparity.
* *(Note: The raw `.graphml` street network and Sentinel-2 satellite outputs are dynamically generated via API in the notebook and are not hosted here due to file size constraints).*

---

## 🚀 How to Run the Code

To reproduce this research:
1. Open `Spatial_Apartheid_Phase1.ipynb` in **Google Colab**.
2. You will need a registered, non-commercial **Google Earth Engine** account. 
3. Run the authentication cell to link your Google Cloud Project.
4. Execute the cells sequentially. The notebook will automatically query OpenStreetMap and Google Earth Engine servers to build the datasets live.

---
*Developed by Kaveri & Vedika for AMD Slingshot.*
