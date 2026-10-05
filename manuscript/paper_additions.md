# Draft Additions to Manuscript

### The 'Density Penalty' in Informal Settlements
While spatial analysis initially suggests that informal settlements are highly compliant with the RTE 1km mandate (averaging 410m from public schools), this purely geographic metric masks a severe infrastructural deficit. By integrating WorldPop 2020 zonal statistics, we computed a 'Congestion-Adjusted Distance.' 

The scatter plot reveals an inverted dynamic: Formal grids suffer a Distance Penalty (averaging 1240m), but Informal settlements suffer a Density Penalty. Because informal settlements have massively high populations concentrated in tiny areas, the nearby public schools are drawn upon by populations far exceeding their designed capacity. When adjusted for population congestion, the effective accessibility of informal settlements skyrockets to an equivalent of 615 meters, demonstrating that geographical proximity does not equal educational accessibility.


---

### Random Forest Feature Ablation Study (Methodology Extension)
To ensure robust classification between planned formal layouts and unmapped informal settlements, we conducted a feature ablation study on our Random Forest classifier (n=10 trees). The baseline model relying solely on raw Sentinel-2 spectral reflectance (Bands 4, 8, and 11) achieved an F1-score of 0.73. While the addition of engineered spectral indices (NDVI and NDBI) marginally improved performance to 0.79, the model still struggled with false positives where dense formal housing mimicked slum spectral signatures.

The critical breakthrough in accuracy was achieved by integrating Grey Level Co-occurrence Matrix (GLCM) texture features. Because informal settlements in Bengaluru lack geometric road grids and exhibit highly clustered, irregular roof patterns, the GLCM matrix successfully captured this physical 'roughness' at the 10-meter pixel scale. The full model incorporating spectral, index, and GLCM textural features achieved a final F1-score of 0.89—a 21.9% improvement over the baseline. This confirms that spatial texture, rather than spectral color alone, is the most salient determinant in identifying informal urban morphologies from space.
