"""Stage 1 - raw dataset inventory & data-quality checks.
Partitions analysed: train (80 trips) and student_test (10 trips). Evaluator data is NOT touched.
Outputs: results/stage1/trip_inventory.csv, results/stage1/partition_summary.csv, results/stage1/quality_issues.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import networkx as nx
from pyproj import Transformer
from config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS
from pathsnap.io import load_trip, list_trips

OUT = RESULTS / "stage1"
OUT.mkdir(parents=True, exist_ok=True)


def local_projector(lon0, lat0):
    """Azimuthal-equidistant projection centred on the trip -> metres, accurate at city scale anywhere on Earth."""
    return Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m", always_xy=True)


def point_to_segments_dist(px, py, ax, ay, bx, by):
    """Min distance from each point to a set of segments (vectorised, chunked)."""
    out = np.empty(len(px))
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    for i in range(0, len(px), 512):
        x = px[i:i + 512, None]; y = py[i:i + 512, None]
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0, 1)
        out[i:i + 512] = np.sqrt((x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2).min(axis=1)
    return out


def inspect(trip_dir: Path, partition: str) -> dict:
    tr = load_trip(trip_dir)
    tid = tr.trip_id
    trk, nodes, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
    r = {"partition": partition, "trip_id": tid}
    for ext in ["track", "nodes", "arcs", "route"]:
        f = trip_dir / f"{tid}.{ext}"
        r[f"{ext}_bytes"] = f.stat().st_size if f.exists() else 0

    # ---- GPS track
    r["n_obs"] = len(trk)
    r["n_nan"] = int(trk.isna().any(axis=1).sum())
    bad = ~trk.lon.between(-180, 180) | ~trk.lat.between(-90, 90) | ((trk.lon == 0) & (trk.lat == 0))
    r["n_invalid_coord"] = int(bad.sum())
    r["n_dup_rows"] = int(trk.duplicated().sum())
    r["n_dup_timestamps"] = int(trk.t.duplicated().sum())
    dt = np.diff(trk.t.to_numpy())
    r["t_start"], r["t_end"] = trk.t.iloc[0], trk.t.iloc[-1]
    r["duration_s"] = r["t_end"] - r["t_start"]
    r["n_nonmonotonic_t"] = int((dt <= 0).sum())
    r["dt_median"] = float(np.median(dt))
    r["pct_dt_eq_1"] = float(np.mean(np.isclose(dt, 1.0)) * 100)
    r["dt_max"] = float(dt.max())
    r["n_gaps_gt5s"] = int((dt > 5).sum())
    r["n_gaps_gt30s"] = int((dt > 30).sum())
    r["non_integer_t"] = int((~np.isclose(trk.t % 1, 0)).sum())

    lon0, lat0 = trk.lon.mean(), trk.lat.mean()
    r["lon_center"], r["lat_center"] = lon0, lat0
    r["lon_min"], r["lon_max"], r["lat_min"], r["lat_max"] = trk.lon.min(), trk.lon.max(), trk.lat.min(), trk.lat.max()
    proj = local_projector(lon0, lat0)
    gx, gy = proj.transform(trk.lon.to_numpy(), trk.lat.to_numpy())
    step = np.hypot(np.diff(gx), np.diff(gy))
    r["track_length_km"] = step.sum() / 1000
    r["bbox_w_km"], r["bbox_h_km"] = (gx.max() - gx.min()) / 1000, (gy.max() - gy.min()) / 1000
    spd = step / np.maximum(dt, 1e-6)
    r["speed_median_mps"] = float(np.median(spd))
    r["speed_p99_mps"] = float(np.percentile(spd, 99))
    r["n_jumps_gt60mps"] = int((spd > 60).sum())        # > 216 km/h -> implausible GPS jump
    r["n_stationary_steps"] = int((step < 0.5).sum())

    # ---- road network
    r["n_nodes"], r["n_arcs"] = len(nodes), len(arcs)
    r["arc_ids_out_of_range"] = int(((arcs.u < 0) | (arcs.u >= len(nodes)) | (arcs.v < 0) | (arcs.v >= len(nodes))).sum())
    r["n_self_loops"] = int((arcs.u == arcs.v).sum())
    r["n_dup_arcs"] = int(arcs.duplicated().sum())
    pairs = set(zip(arcs.u, arcs.v))
    r["pct_arcs_with_reverse"] = 100 * np.mean([(v, u) in pairs for u, v in pairs])
    G = nx.DiGraph(); G.add_nodes_from(range(len(nodes))); G.add_edges_from(zip(arcs.u, arcs.v))
    deg = np.array([d for _, d in G.degree()])
    r["n_isolated_nodes"] = int((deg == 0).sum())
    wcc = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    r["n_weak_components"] = len(wcc)
    r["largest_wcc_frac"] = wcc[0] / len(nodes)
    scc = max(nx.strongly_connected_components(G), key=len)
    r["largest_scc_frac"] = len(scc) / len(nodes)
    nx_, ny_ = proj.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    alen = np.hypot(nx_[arcs.v] - nx_[arcs.u], ny_[arcs.v] - ny_[arcs.u])
    r["arc_len_median_m"], r["arc_len_max_m"] = float(np.median(alen)), float(alen.max())
    r["n_zero_len_arcs"] = int((alen < 0.01).sum())
    r["net_bbox_w_km"], r["net_bbox_h_km"] = (nx_.max() - nx_.min()) / 1000, (ny_.max() - ny_.min()) / 1000

    # ---- reference route
    if route is not None:
        r["route_len_arcs"] = len(route)
        r["route_unique_arcs"] = len(np.unique(route))
        r["route_out_of_range"] = int(((route < 0) | (route >= len(arcs))).sum())
        u, v = arcs.u.to_numpy()[route], arcs.v.to_numpy()[route]
        r["route_breaks"] = int((v[:-1] != u[1:]).sum())          # consecutive arcs not head-to-tail
        r["route_consecutive_repeats"] = int((route[:-1] == route[1:]).sum())
        r["route_length_km"] = alen[route].sum() / 1000
        # GPS noise w.r.t. reference geometry
        d = point_to_segments_dist(gx, gy, nx_[u], ny_[u], nx_[v], ny_[v])
        r["gps_to_route_median_m"] = float(np.median(d))
        r["gps_to_route_p95_m"] = float(np.percentile(d, 95))
        r["gps_to_route_max_m"] = float(d.max())
        r["pct_gps_within_10m"] = float(np.mean(d < 10) * 100)
        # how many arcs fall near the GPS (candidate-set size hint): arcs within 30 m of median GPS pt
        r["route_len_vs_track_len"] = r["route_length_km"] / max(r["track_length_km"], 1e-9)
    return r


def main():
    rows = []
    for part, d in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for td in list_trips(d):
            rows.append(inspect(td, part))
            print(f"{part:12s} {td.name}  obs={rows[-1]['n_obs']:5d}  nodes={rows[-1]['n_nodes']:6d}  arcs={rows[-1]['n_arcs']:6d}  route={rows[-1].get('route_len_arcs')}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_inventory.csv", index=False)

    agg = df.groupby("partition").agg(
        trips=("trip_id", "count"), total_obs=("n_obs", "sum"),
        obs_min=("n_obs", "min"), obs_median=("n_obs", "median"), obs_max=("n_obs", "max"),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
        nan_rows=("n_nan", "sum"), invalid_coords=("n_invalid_coord", "sum"),
        nonmonotonic_t=("n_nonmonotonic_t", "sum"), gaps_gt5s=("n_gaps_gt5s", "sum"), gaps_gt30s=("n_gaps_gt30s", "sum"),
        pct_dt_eq_1=("pct_dt_eq_1", "mean"), route_breaks=("route_breaks", "sum"),
        gps_route_median_m=("gps_to_route_median_m", "median"), gps_route_p95_m=("gps_to_route_p95_m", "median"),
        track_km_median=("track_length_km", "median"))
    agg.to_csv(OUT / "partition_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(agg.T)
    print(df.describe().T[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
