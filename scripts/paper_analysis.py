"""
Reproducible analysis behind every number and figure in manuscript/ieee_manuscript.tex.

Uses only real inputs already in data/:
  - bengaluru_walk.graphml               OSM pedestrian network (OSMnx)
  - bengaluru_schools.geojson            all OSM amenity=school features
  - bengaluru_schools_corrected.geojson  government-school subset produced by the notebook ("broad" filter)
  - informal_samples / formal_samples    points sampled from the Sentinel-2 Random Forest classification
  - blr_slums.json                       KSDB slum polygons
  - KSDB_Slums_WorldPop_Density.csv      WorldPop 2020 (100 m) population summed per slum polygon

Two government-school sets are analysed:
  - "curated" (primary): names carrying an explicit government keyword, excluding colleges and
    private schools whose names only end in a BBMP ward address.
  - "broad" (notebook): the 271-school notebook filter, which also matched any name containing
    "Public" and therefore includes private schools (e.g. "Delhi Public School"-style names).

Writes tables to results/paper_stats/ and figures to results/fig_*.png.
Run from the project root:  .venv/bin/python scripts/paper_analysis.py
"""
import heapq
import json
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy import stats

DATA = "data"
OUT = "results"
STATS = os.path.join(OUT, "paper_stats")
os.makedirs(STATS, exist_ok=True)

UTM = "EPSG:32643"
RTE_M = 1000
FORMAL_C, INFORMAL_C, NEUTRAL_C = "#1F4E9C", "#D1453B", "#7A7A7A"
GOVT_KW = r"govt|government|bbmp|ghps|glps|ghs|gulps|kgbv|corporation|municipal"
NOT_PRIMARY_KW = r"college|\bpuc\b|\bpu\b"
COL = 3.5  # IEEE single-column width (inches)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
rng = np.random.default_rng(0)
t0 = time.time()


def save_json(obj, name):
    with open(os.path.join(STATS, name), "w") as fh:
        json.dump(obj, fh, indent=2, default=float)


# ====================== inputs ======================
print("Loading graph...")
G = ox.load_graphml(os.path.join(DATA, "bengaluru_walk.graphml"))
edge_len = np.array([d["length"] for _, _, d in G.edges(data=True)])
print(f"  {len(G.nodes):,} nodes, {len(G.edges):,} edges ({time.time()-t0:.0f}s)")

all_schools = gpd.read_file(os.path.join(DATA, "bengaluru_schools.geojson"))
broad = gpd.read_file(os.path.join(DATA, "bengaluru_schools_corrected.geojson"))
inf_pts = gpd.read_file(os.path.join(DATA, "informal_samples.geojson"))
for_pts = gpd.read_file(os.path.join(DATA, "formal_samples.geojson"))
slums = gpd.read_file(os.path.join(DATA, "blr_slums.json")).to_crs(4326)
pop = pd.read_csv(os.path.join(DATA, "KSDB_Slums_WorldPop_Density.csv"))[["OBJECTID", "sum"]]
slums = slums.merge(pop.rename(columns={"sum": "pop"}), on="OBJECTID", how="left")
slums_utm = slums.to_crs(UTM)
slums["area_ha"] = slums_utm.geometry.area.values / 1e4
slums["density_per_ha"] = slums["pop"] / slums["area_ha"]
slums["notified"] = np.where(slums["Slum_Type"].str.lower().str.replace(r"[^a-z]", "", regex=True)
                             .str.startswith("not"), "Notified", "Non-notified")


def centroids(gdf):
    return gdf.to_crs(UTM).geometry.centroid.to_crs(4326)


all_c, broad_c = centroids(all_schools), centroids(broad)

# --- curated government set ---
names = broad["name"].fillna("").str.lower()
head = names.str.split(",").str[0]  # keyword must appear before any address suffix
kw_any = names.str.contains(GOVT_KW)
kw_head = head.str.contains(GOVT_KW)
not_primary = names.str.contains(NOT_PRIMARY_KW)
curated_mask = (kw_head & ~not_primary).values
broad[["name"]].assign(curated=curated_mask, keyword_any=kw_any.values, college=not_primary.values) \
    .to_csv(os.path.join(STATS, "govt_school_classification.csv"), index=False)

school_stats = {
    "all_osm_schools": len(all_schools),
    "all_polygon": int((all_schools.geom_type != "Point").sum()),
    "all_point": int((all_schools.geom_type == "Point").sum()),
    "all_unnamed": int(all_schools["name"].isna().sum()),
    "broad": len(broad),
    "broad_public_only": int((names.str.contains("public") & ~kw_any).sum()),
    "broad_no_keyword": int((~kw_any).sum()),
    "keyword_any": int(kw_any.sum()),
    "address_suffix_only": int((kw_any & ~kw_head).sum()),
    "colleges_excluded": int((kw_head & not_primary).sum()),
    "curated": int(curated_mask.sum()),
    "curated_primary_named": int(names[curated_mask].str.contains(r"primary|lps|hps|lower").sum()),
}
graph_stats = {"nodes": len(G.nodes), "edges": len(G.edges),
               "edge_km_directed": float(edge_len.sum() / 1000),
               "mean_edge_m": float(edge_len.mean()), "median_edge_m": float(np.median(edge_len))}

# ====================== routing primitives ======================


def snap(points):
    nodes, dist = ox.distance.nearest_nodes(G, X=points.x.values, Y=points.y.values, return_dist=True)
    return np.asarray(nodes), np.asarray(dist)


def labelled_dijkstra(sources):
    """Multi-source Dijkstra that also records which source each node is nearest to."""
    dist, owner, heap = {}, {}, []
    for i, s in enumerate(sources):
        if s not in dist:
            dist[s], owner[s] = 0.0, i
            heap.append((0.0, s, i))
    heapq.heapify(heap)
    while heap:
        d, u, src = heapq.heappop(heap)
        if d > dist.get(u, np.inf):
            continue
        for v, edata in G._adj[u].items():
            nd = d + min(e.get("length", 0.0) for e in edata.values())
            if nd < dist.get(v, np.inf):
                dist[v], owner[v] = nd, src
                heapq.heappush(heap, (nd, v, src))
    return dist, owner


def euclid_nearest(points_wgs, school_c):
    s = school_c.to_crs(UTM)
    sxy = np.c_[s.x.values, s.y.values]
    p = points_wgs.to_crs(UTM)
    xy = np.c_[p.x.values, p.y.values]
    return np.sqrt(((xy[:, None, :] - sxy[None, :, :]) ** 2).sum(-1)).min(1)


inf_nodes, inf_snap = snap(inf_pts.geometry)
for_nodes, for_snap = snap(for_pts.geometry)
slum_c = slums_utm.geometry.centroid.to_crs(4326)
slum_nodes, slum_snap = snap(slum_c)


def describe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return {"n": len(x), "mean": x.mean(), "sd": x.std(ddof=1), "median": np.median(x),
            "q1": np.percentile(x, 25), "q3": np.percentile(x, 75), "p90": np.percentile(x, 90),
            "min": x.min(), "max": x.max(),
            "pct_le_1km": 100 * (x <= 1000).mean(), "pct_gt_2km": 100 * (x > 2000).mean()}


def gini(x):
    x = np.asarray(x, float)
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean()))


def analyse(tag, school_c, school_names):
    """All accessibility metrics for one government-school set."""
    nodes, snapd = snap(school_c)
    dist, owner = labelled_dijkstra(list(nodes))

    def lookup(ns):
        return np.array([dist.get(n, np.nan) for n in ns])

    samples = pd.concat([
        pd.DataFrame({"zone": "Informal", "node": inf_nodes, "snap_m": inf_snap,
                      "net_m": lookup(inf_nodes), "euclid_m": euclid_nearest(inf_pts.geometry, school_c)}),
        pd.DataFrame({"zone": "Formal", "node": for_nodes, "snap_m": for_snap,
                      "net_m": lookup(for_nodes), "euclid_m": euclid_nearest(for_pts.geometry, school_c)}),
    ], ignore_index=True)
    samples["detour"] = samples.net_m / samples.euclid_m.clip(lower=50)
    samples.to_csv(os.path.join(STATS, f"{tag}_sample_distances.csv"), index=False)

    desc = pd.DataFrame([{"zone": z, "metric": m, **describe(samples.loc[samples.zone == z, m])}
                         for z in ["Formal", "Informal"] for m in ["net_m", "euclid_m", "detour"]])
    desc.to_csv(os.path.join(STATS, f"{tag}_descriptives.csv"), index=False)

    f = samples.loc[samples.zone == "Formal", "net_m"].dropna().values
    i = samples.loc[samples.zone == "Informal", "net_m"].dropna().values
    U, p_mw = stats.mannwhitneyu(f, i, alternative="two-sided")
    boot = [rng.choice(f, len(f)).mean() - rng.choice(i, len(i)).mean() for _ in range(5000)]
    ks = stats.ks_2samp(f, i)
    chi2, p_chi, _, _ = stats.chi2_contingency(pd.crosstab(samples.zone, samples.net_m <= RTE_M))
    tests = {
        "mean_formal": f.mean(), "mean_informal": i.mean(),
        "mean_diff_m": f.mean() - i.mean(),
        "boot_ci_low": np.percentile(boot, 2.5), "boot_ci_high": np.percentile(boot, 97.5),
        "mannwhitney_U": U, "mannwhitney_p": p_mw, "cliffs_delta": 2 * U / (len(f) * len(i)) - 1,
        "ks_D": ks.statistic, "ks_p": ks.pvalue, "chi2": chi2, "chi2_p": p_chi,
        "spearman_net_euclid": stats.spearmanr(samples.net_m, samples.euclid_m, nan_policy="omit").statistic,
        "crowflies_compliant_pct": 100 * (samples.euclid_m <= RTE_M).mean(),
        "network_compliant_pct": 100 * (samples.net_m <= RTE_M).mean(),
        "false_compliant_pct": 100 * ((samples.euclid_m <= RTE_M) & (samples.net_m > RTE_M)).mean(),
        "false_compliant_share_of_crowflies_pct":
            100 * ((samples.euclid_m <= RTE_M) & (samples.net_m > RTE_M)).sum() / (samples.euclid_m <= RTE_M).sum(),
    }
    thr = np.arange(250, 3001, 250)
    sweep = pd.DataFrame({"threshold_m": thr,
                          "formal_pct": [(f <= t).mean() * 100 for t in thr],
                          "informal_pct": [(i <= t).mean() * 100 for t in thr]})
    sweep.to_csv(os.path.join(STATS, f"{tag}_threshold_sweep.csv"), index=False)

    sl = slums.drop(columns="geometry").copy()
    sl["net_m"] = lookup(slum_nodes)
    sl["euclid_m"] = euclid_nearest(slum_c, school_c)
    sl["nearest_school"] = [owner.get(n, -1) for n in slum_nodes]
    sl.to_csv(os.path.join(STATS, f"{tag}_slum_distances.csv"), index=False)
    sp = sl.dropna(subset=["net_m", "pop"])

    def popsummary(d):
        return pd.Series({
            "slums": len(d), "population": d["pop"].sum(), "median_dist_m": d.net_m.median(),
            "pop_weighted_dist_m": np.average(d.net_m, weights=d["pop"]),
            "pct_slums_le_1km": 100 * (d.net_m <= RTE_M).mean(),
            "pct_pop_le_1km": 100 * d.loc[d.net_m <= RTE_M, "pop"].sum() / d["pop"].sum()})

    rho = stats.spearmanr(sp["pop"], sp.net_m)
    rho_d = stats.spearmanr(sp.density_per_ha, sp.net_m)
    slum_summary = {**popsummary(sp).to_dict(),
                    "mean_dist_m": sp.net_m.mean(), "pct_slums_gt_2km": 100 * (sp.net_m > 2000).mean(),
                    "spearman_pop_dist": rho.statistic, "spearman_pop_dist_p": rho.pvalue,
                    "spearman_density_dist": rho_d.statistic, "spearman_density_dist_p": rho_d.pvalue}
    zone_tbl = sp.groupby("Zone_Name").apply(popsummary, include_groups=False).sort_values("pop_weighted_dist_m")
    zone_tbl.to_csv(os.path.join(STATS, f"{tag}_zone_table.csv"))
    notif_tbl = sp.groupby("notified").apply(popsummary, include_groups=False)
    notif_tbl.to_csv(os.path.join(STATS, f"{tag}_notified_table.csv"))
    a = sp.loc[sp.notified == "Non-notified", "net_m"]
    b = sp.loc[sp.notified == "Notified", "net_m"]
    Un, pn = stats.mannwhitneyu(a, b, alternative="two-sided")
    slum_summary.update({"notified_U": Un, "notified_p": pn, "notified_cliffs": 2 * Un / (len(a) * len(b)) - 1})

    load = sp.groupby("nearest_school")["pop"].agg(["sum", "count"]).rename(columns={"sum": "slum_pop", "count": "slums"})
    load = load.reindex(range(len(school_c)), fill_value=0)
    load["name"] = list(school_names)
    load.sort_values("slum_pop", ascending=False).to_csv(os.path.join(STATS, f"{tag}_catchment_load.csv"))
    served = load[load.slum_pop > 0]
    catch = {"schools": len(load), "with_slum_catchment": int(len(served)),
             "without": int((load.slum_pop == 0).sum()),
             "median_load_served": served.slum_pop.median(), "mean_load_served": served.slum_pop.mean(),
             "max_load": load.slum_pop.max(),
             "top10_share_pct": 100 * load.slum_pop.nlargest(10).sum() / load.slum_pop.sum(),
             "gini": gini(load.slum_pop.values)}

    out = {"n_schools": len(school_c), "snap_median_m": float(np.median(snapd)), "tests": tests,
           "slums": slum_summary, "catchment": catch}
    save_json(out, f"{tag}_summary.json")
    return out, samples, sp, sweep, zone_tbl, notif_tbl, load


print("Routing (curated)...")
cur, cur_samples, cur_sp, cur_sweep, cur_zone, cur_notif, cur_load = analyse(
    "curated", broad_c[curated_mask], broad["name"][curated_mask].fillna("(unnamed)"))
print("Routing (broad)...")
brd, *_ = analyse("broad", broad_c, broad["name"].fillna("(unnamed)"))

# proximity to the remaining (non-government) OSM schools
nongovt_mask = ~all_schools["id"].astype(str).isin(set(broad["id"].astype(str)[curated_mask]))
ng_nodes, _ = snap(all_c[nongovt_mask.values])
dist_ng = nx.multi_source_dijkstra_path_length(G, set(ng_nodes), weight="length")
nongovt = {z: describe([dist_ng.get(n, np.nan) for n in ns])
           for z, ns in [("Formal", for_nodes), ("Informal", inf_nodes)]}
nongovt["n_schools"] = int(nongovt_mask.sum())

# classifier agreement with KSDB polygons (proxy validation of the RF "informal" class)
def share_within(points, geoms):
    j = gpd.sjoin(points, gpd.GeoDataFrame(geometry=geoms), predicate="within", how="left")
    return 100 * j["index_right"].notna().groupby(level=0).any().mean()


agree = {"informal_n": len(inf_pts), "formal_n": len(for_pts)}
for label, geoms in [("inside", slums.geometry),
                     ("within100", slums_utm.buffer(100).to_crs(4326)),
                     ("within250", slums_utm.buffer(250).to_crs(4326)),
                     ("within500", slums_utm.buffer(500).to_crs(4326))]:
    agree[f"informal_{label}_pct"] = share_within(inf_pts, geoms)
    agree[f"formal_{label}_pct"] = share_within(for_pts, geoms)

slum_desc = {"n": len(slums), "total_pop": slums["pop"].sum(),
             "median_area_ha": slums.area_ha.median(), "median_pop": slums["pop"].median(),
             "median_density_per_ha": slums.density_per_ha.median(),
             "notified": int((slums.notified == "Notified").sum()),
             "non_notified": int((slums.notified == "Non-notified").sum()),
             "zones": int(slums.Zone_Name.nunique())}

summary = {"graph": graph_stats, "schools": school_stats, "classifier_agreement": agree,
           "slum_inputs": slum_desc, "nongovt_proximity": nongovt,
           "sample_snap_median_m": {"formal": float(np.median(for_snap)), "informal": float(np.median(inf_snap))},
           "slum_snap_median_m": float(np.median(slum_snap)),
           "curated": cur, "broad": brd, "runtime_s": time.time() - t0}
save_json(summary, "summary.json")
print(json.dumps(summary, indent=1, default=lambda v: round(float(v), 3)))
print(cur_zone.round(1).to_string())
print(cur_notif.round(1).to_string())
print(cur_load.sort_values("slum_pop", ascending=False).head(8).round(0).to_string())
print(pd.read_csv(os.path.join(STATS, "curated_descriptives.csv")).round(1).to_string())

# ====================== figures (curated set) ======================
samples, sp, sweep, zone_tbl, load = cur_samples, cur_sp, cur_sweep, cur_zone, cur_load


def legend_handles():
    return [Line2D([], [], marker="s", ls="", color=FORMAL_C, label="Planned (formal)"),
            Line2D([], [], marker="o", ls="", color=INFORMAL_C, label="Informal")]


# mean distance with bootstrap CI, both school sets
fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 1.8), sharey=True)
for ax, (tag, df) in zip(axes, [("Curated govt. schools (n=%d)" % cur["n_schools"], cur_samples),
                                ("Broad notebook filter (n=%d)" % brd["n_schools"],
                                 pd.read_csv(os.path.join(STATS, "broad_sample_distances.csv")))]):
    for k, (zone, c) in enumerate([("Formal", FORMAL_C), ("Informal", INFORMAL_C)]):
        x = df.loc[df.zone == zone, "net_m"].dropna().values / 1000
        bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.barh(k, x.mean(), color=c, height=0.55)
        ax.errorbar(x.mean(), k, xerr=[[x.mean() - lo], [hi - x.mean()]], color="black", capsize=3, lw=1)
        ax.text(hi + 0.07, k, f"{x.mean():.2f} km (median {np.median(x):.2f})", va="center", fontsize=8)
    ax.axvline(1, color=NEUTRAL_C, ls="--", lw=1)
    ax.text(1.03, -0.45, "RTE 1 km", color=NEUTRAL_C, fontsize=7)
    ax.set_yticks([0, 1], ["Planned\n(formal)", "Informal"])
    ax.set_ylim(1.45, -0.62)
    ax.set_xlim(0, 4.3)
    ax.set_title(tag)
    ax.set_xlabel("Mean network walking distance (km)")
    ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig_mean_distance.png"))
plt.close(fig)

# ECDF
fig, ax = plt.subplots(figsize=(COL, 2.4))
for zone, c, ls in [("Formal", FORMAL_C, "-"), ("Informal", INFORMAL_C, "--")]:
    x = np.sort(samples.loc[samples.zone == zone, "net_m"].dropna().values / 1000)
    ax.step(x, np.arange(1, len(x) + 1) / len(x) * 100, where="post", color=c, lw=1.6, ls=ls,
            label=f"{'Planned (formal)' if zone == 'Formal' else 'Informal'} (n={len(x)})")
ax.axvline(1, color=NEUTRAL_C, ls=":", lw=1)
ax.text(1.05, 92, "RTE 1 km", color=NEUTRAL_C, fontsize=7)
ax.set_xlabel("Network walking distance to nearest govt. school (km)")
ax.set_ylabel("Cumulative share of points (%)")
ax.set_xlim(0, np.nanpercentile(samples.net_m, 99) / 1000)
ax.set_ylim(0, 100)
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.06), frameon=False)
fig.savefig(os.path.join(OUT, "fig_distance_ecdf.png"))
plt.close(fig)

# Euclidean vs network
fig, ax = plt.subplots(figsize=(COL, 3.0))
for zone, c, m in [("Formal", FORMAL_C, "s"), ("Informal", INFORMAL_C, "o")]:
    s = samples[samples.zone == zone]
    ax.scatter(s.euclid_m / 1000, s.net_m / 1000, s=7, c=c, marker=m, alpha=0.55, lw=0)
lim = np.nanpercentile(samples.net_m, 99.5) / 1000
ax.plot([0, lim], [0, lim], color=NEUTRAL_C, lw=1, label="Network = straight line")
ax.axhline(1, color=NEUTRAL_C, ls=":", lw=0.8)
ax.axvline(1, color=NEUTRAL_C, ls=":", lw=0.8)
ax.fill_between([0, 1], 1, lim, color="#F2C14E", alpha=0.18, lw=0)
ax.text(0.05, lim * 0.97, "Crow-flies compliant,\nnetwork non-compliant", fontsize=7, va="top")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Straight-line (Euclidean) distance (km)")
ax.set_ylabel("Network walking distance (km)")
ax.legend(handles=legend_handles() + [ax.get_lines()[0]], loc="upper center",
          bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=7)
fig.savefig(os.path.join(OUT, "fig_euclid_vs_network.png"))
plt.close(fig)

# threshold sweep
fig, ax = plt.subplots(figsize=(COL, 2.2))
ax.plot(sweep.threshold_m / 1000, sweep.formal_pct, color=FORMAL_C, marker="s", ms=4, lw=1.6, label="Planned (formal)")
ax.plot(sweep.threshold_m / 1000, sweep.informal_pct, color=INFORMAL_C, marker="o", ms=4, lw=1.6, ls="--", label="Informal")
ax.axvline(1, color=NEUTRAL_C, ls=":", lw=1)
ax.set_xlabel("Accessibility threshold (km)")
ax.set_ylabel("Points within threshold (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="lower right")
fig.savefig(os.path.join(OUT, "fig_threshold_sweep.png"))
plt.close(fig)

# slum population vs distance
fig, ax = plt.subplots(figsize=(COL, 2.6))
for lab, mk, c in [("Notified", "o", INFORMAL_C), ("Non-notified", "^", "#8C2D19")]:
    d = sp[sp.notified == lab]
    ax.scatter(d["pop"], d.net_m / 1000, s=10, marker=mk, c=c, alpha=0.55, lw=0, label=lab)
ax.set_xscale("log")
ax.axhline(1, color=NEUTRAL_C, ls="--", lw=1)
ax.text(4000, 0.75, "RTE 1 km", color=NEUTRAL_C, fontsize=7, ha="right")
ax.set_xlabel("Slum population, WorldPop 2020 (log scale)")
ax.set_ylabel("Network distance to nearest\ngovt. school (km)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.savefig(os.path.join(OUT, "fig_slum_pop_vs_distance.png"))
plt.close(fig)

# catchment load
fig, ax = plt.subplots(figsize=(COL, 2.2))
vals = load.slum_pop.sort_values(ascending=False).values
vals = vals[vals > 0]
ax.bar(np.arange(1, len(vals) + 1), vals / 1000, color=INFORMAL_C, width=0.8)
ax.set_xlabel("Government schools ranked by assigned slum population")
ax.set_ylabel("Slum residents in network\ncatchment (thousands)")
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(OUT, "fig_catchment_load.png"))
plt.close(fig)

# administrative zone bar
zt = zone_tbl.sort_values("pop_weighted_dist_m")
fig, ax = plt.subplots(figsize=(COL, 2.4))
ax.barh(zt.index, zt.pop_weighted_dist_m / 1000, color=INFORMAL_C, height=0.6)
ax.axvline(1, color=NEUTRAL_C, ls="--", lw=1)
for k, v in enumerate(zt.pop_weighted_dist_m / 1000):
    ax.text(v + 0.03, k, f"{v:.2f}", va="center", fontsize=7)
ax.set_xlabel("Population-weighted distance to govt. school (km)")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig_zone_distance.png"))
plt.close(fig)

# study area
fig, ax = plt.subplots(figsize=(COL, 3.6))
slums.plot(ax=ax, color="#F2C14E", edgecolor="#B8860B", lw=0.2)
for_pts.plot(ax=ax, color=FORMAL_C, marker="s", markersize=1.5, alpha=0.6, linewidth=0)
inf_pts.plot(ax=ax, color=INFORMAL_C, marker="o", markersize=1.5, alpha=0.8, linewidth=0)
broad_c[curated_mask].plot(ax=ax, color="black", marker="^", markersize=6, linewidth=0)
ax.legend(handles=[Patch(color="#F2C14E", label="KSDB slum polygons"),
                   Line2D([], [], marker="s", ls="", color=FORMAL_C, ms=4, label="Planned samples (RF)"),
                   Line2D([], [], marker="o", ls="", color=INFORMAL_C, ms=4, label="Informal samples (RF)"),
                   Line2D([], [], marker="^", ls="", color="black", ms=5, label="Govt. schools (curated)")],
          loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7)
ax.set_axis_off()
fig.savefig(os.path.join(OUT, "fig_study_area.png"))
plt.close(fig)

# vulnerability-map screenshots: crop away browser chrome and the macOS dock
for name in ["vulnerability_map_slum_zone_0_55km", "vulnerability_map_planned_zone_2_35km"]:
    Image.open(os.path.join(OUT, name + ".jpg")).crop((14, 62, 1020, 607)).save(
        os.path.join(OUT, name + "_cropped.png"))

print(f"All done in {time.time()-t0:.0f}s")

# ---------- vulnerability maps from the curated school set (static figure + interactive HTML) ----------
import contextily as cx
import folium

BANDS = [(RTE_M, "#2E7D32", "$\\leq$1 km (compliant)"), (2000, "#F29F05", "1–2 km"), (np.inf, "#C62828", ">2 km (critical)")]


def band_colour(d):
    for lim, c, _ in BANDS:
        if d <= lim:
            return c
    return NEUTRAL_C


pts = pd.concat([inf_pts.assign(zone="Informal"), for_pts.assign(zone="Formal")], ignore_index=True)
pts["net_m"] = cur_samples["net_m"].values
pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=4326).to_crs(3857)
sl_pts = gpd.GeoDataFrame(sp, geometry=slum_c.loc[sp.index].values, crs=4326).to_crs(3857)
sch = broad_c[curated_mask].to_crs(3857)

fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 3.9))
for ax, title in zip(axes, ["(a) Classified sample locations", "(b) KSDB slums (marker size = population)"]):
    ax.set_title(title)
for zone, mk in [("Formal", "s"), ("Informal", "o")]:
    d = pts[pts.zone == zone]
    axes[0].scatter(d.geometry.x, d.geometry.y, c=[band_colour(v) for v in d.net_m], marker=mk, s=9,
                    lw=0.3, edgecolors="white", zorder=3)
axes[1].scatter(sl_pts.geometry.x, sl_pts.geometry.y, c=[band_colour(v) for v in sl_pts.net_m],
                s=np.clip(np.sqrt(sl_pts["pop"]) * 0.9, 3, 70), lw=0.3, edgecolors="white", alpha=0.9, zorder=3)
xmin, ymin, xmax, ymax = pts.total_bounds
for ax in axes:
    ax.scatter(sch.x, sch.y, marker="^", c="black", s=14, lw=0, zorder=4)
    ax.set_xlim(xmin - 1500, xmax + 1500)
    ax.set_ylim(ymin - 1500, ymax + 1500)
    cx.add_basemap(ax, source=cx.providers.Esri.WorldGrayCanvas, attribution_size=5)
    ax.set_axis_off()
handles = [Line2D([], [], marker="o", ls="", color=c, ms=5, label=l) for _, c, l in BANDS] + [
    Line2D([], [], marker="^", ls="", color="black", ms=5, label="Govt. school (curated)"),
    Line2D([], [], marker="s", ls="", color=NEUTRAL_C, ms=4, label="Planned sample"),
    Line2D([], [], marker="o", ls="", color=NEUTRAL_C, ms=4, label="Informal sample")]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "fig_vulnerability_map.png"))
plt.close(fig)

m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="OpenStreetMap")
pts_w = pts.to_crs(4326)
for _, r in pts_w.iterrows():
    folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color=band_colour(r.net_m), fill=True,
                        fill_opacity=0.8, popup=f"{'Planned' if r.zone == 'Formal' else 'Informal'} sample<br>"
                                                f"Walk to govt. school: {r.net_m / 1000:.2f} km").add_to(m)
for (_, r), c in zip(sp.iterrows(), slum_c.loc[sp.index]):
    folium.CircleMarker([c.y, c.x], radius=float(np.clip(np.sqrt(r["pop"]) / 6, 2, 12)), color=band_colour(r.net_m),
                        weight=1, fill=True, fill_opacity=0.5,
                        popup=f"{r.Slum_Name} ({r.notified})<br>Population ~{r['pop']:.0f}<br>"
                              f"Walk: {r.net_m / 1000:.2f} km").add_to(m)
for c, n in zip(broad_c[curated_mask], broad["name"][curated_mask].fillna("Government school")):
    folium.RegularPolygonMarker([c.y, c.x], number_of_sides=3, radius=5, color="black", fill=True,
                                fill_opacity=1, popup=n).add_to(m)
m.save(os.path.join(OUT, "vulnerability_map_curated.html"))
print("vulnerability maps written")

# ---------- network Voronoi catchment map and zone distance distributions (curated set) ----------
cur_nodes, _ = snap(broad_c[curated_mask])
dist_c, owner_c = labelled_dijkstra(list(cur_nodes))
node_ids = np.fromiter(dist_c.keys(), dtype=np.int64)
nx_ = np.array([G.nodes[n]["x"] for n in node_ids])
ny_ = np.array([G.nodes[n]["y"] for n in node_ids])
nd_ = np.array([dist_c[n] for n in node_ids])
no_ = np.array([owner_c[n] for n in node_ids])
xy = gpd.GeoSeries(gpd.points_from_xy(nx_, ny_), crs=4326).to_crs(3857)
cmap = plt.get_cmap("tab20")
cols = cmap(no_ % 20)
cols[nd_ > RTE_M, 3] = 0.08  # fade nodes beyond the 1 km RTE neighbourhood

fig, ax = plt.subplots(figsize=(COL * 2, 6.2))
ax.scatter(xy.x, xy.y, c=cols, s=0.4, lw=0, rasterized=True)
sl_m = gpd.GeoDataFrame(sp, geometry=slum_c.loc[sp.index].values, crs=4326).to_crs(3857)
ax.scatter(sl_m.geometry.x, sl_m.geometry.y, s=np.clip(np.sqrt(sl_m["pop"]) * 0.8, 2, 60), facecolors="none",
           edgecolors="black", lw=0.5, zorder=3)
sc_m = broad_c[curated_mask].to_crs(3857)
ax.scatter(sc_m.x, sc_m.y, marker="^", c="black", s=18, lw=0, zorder=4)
ax.set_axis_off()
ax.set_aspect("equal")
ax.legend(handles=[Patch(color=cmap(3), label="Street within 1 km walk of a govt. school\n(colour = which school)"),
                   Patch(color=(0.6, 0.6, 0.6, 0.25), label="Street beyond 1 km"),
                   Line2D([], [], marker="^", ls="", color="black", ms=6, label="Govt. school (curated)"),
                   Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=6, label="KSDB slum (size = population)")],
          loc="lower left", frameon=True, framealpha=0.9, fontsize=7)
fig.savefig(os.path.join(OUT, "fig_catchment_map.png"))
plt.close(fig)

order = zone_tbl.sort_values("pop_weighted_dist_m").index
fig, ax = plt.subplots(figsize=(COL, 2.8))
data = [sp.loc[sp.Zone_Name == z, "net_m"].values / 1000 for z in order]
bp = ax.boxplot(data, vert=False, tick_labels=list(order), widths=0.55, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", ms=2, mfc=NEUTRAL_C, mec="none"),
                medianprops=dict(color="black", lw=1.2))
for b in bp["boxes"]:
    b.set(facecolor=INFORMAL_C, alpha=0.75, lw=0.6)
ax.axvline(1, color=NEUTRAL_C, ls="--", lw=1)
ax.set_xlabel("Network distance from slum to nearest govt. school (km)")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "fig_zone_boxplot.png"))
plt.close(fig)
print("catchment map and zone boxplot written")
