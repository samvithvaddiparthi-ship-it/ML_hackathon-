"""Stage B1 - Exploratory data analysis for Report section 2.2.

Partitions analysed: train (80 trips) and student_test (10 trips). The evaluator-test partition is
never read. Every figure title states the partition it was computed on.

Run:  PATHSNAP_DATA="/path/to/Student folder" .venv/bin/python eda/02_eda.py
Outputs: results/eda/*.png, results/eda/*.csv, results/eda/eda_summary.md, results/maps/*.html (+ .png)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import TRAIN_DIR, STUDENT_TEST_DIR, RESULTS
from pathsnap.io import load_trip, list_trips
from pathsnap.geo import make_projector, RoadNet
from viz.folium_maps import make_trip_map, network_segments_near

OUT = RESULTS / "eda"
MAPS = RESULTS / "maps"
OUT.mkdir(parents=True, exist_ok=True)
MAPS.mkdir(parents=True, exist_ok=True)
DPI = 150
PARTS = [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]
PCOL = {"train": "#1f6fd1", "student_test": "#f28c00"}
plt.rcParams.update({"figure.dpi": DPI, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})

# trips used for the per-trip figures (chosen for variety: small/large network, clean/noisy, both partitions)
TRAJ_TRIPS = [("train", "00000001"), ("train", "00000032"), ("train", "00000075"), ("student_test", "00000052")]
NET_TRIPS = [("train", "00000046"), ("train", "00000075"), ("student_test", "00000052")]
MAP_TRIPS = [("train", "00000075"), ("student_test", "00000052")]
# trips whose reference route ends before the GPS does (results/stage1/quality_issues.md, item 4)
TRUNCATED_ROUTE = {"00000002", "00000033", "00000054"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("  wrote", name)


def dist_to_route(gx, gy, net: RoadNet, route, step=2.0):
    """Distance (m) of each GPS point to the reference-route polyline, via a KD-tree on the route
    densified every `step` m (error <= step/2, far below GPS noise)."""
    ax, ay, dx, dy, L = net.ax[route], net.ay[route], net.dx[route], net.dy[route], net.length[route]
    n = np.maximum(np.ceil(L / step).astype(int), 1) + 1
    rep = np.repeat(np.arange(len(route)), n)
    frac = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    pts = np.c_[ax[rep] + frac * dx[rep], ay[rep] + frac * dy[rep]]
    d, _ = cKDTree(pts).query(np.c_[gx, gy])
    return d


# ------------------------------------------------------------------ pass over all trips
def collect():
    per_trip, per_obs = [], []
    for part, d in PARTS:
        for td in list_trips(d):
            tr = load_trip(td, with_route=True)
            trk = tr.track
            proj = make_projector(trk.lon.mean(), trk.lat.mean())
            gx, gy = proj.transform(trk.lon.to_numpy(), trk.lat.to_numpy())
            dt = np.diff(trk.t.to_numpy())
            step = np.hypot(np.diff(gx), np.diff(gy))
            spd = step / np.maximum(dt, 1e-6)
            net = RoadNet(tr.nodes, tr.arcs, proj)
            dr = dist_to_route(gx, gy, net, tr.route)
            sizes = {ext: (td / f"{tr.trip_id}.{ext}").stat().st_size for ext in ["track", "nodes", "arcs", "route"]}
            nan_cols = {f"nan_{c}": int(trk[c].isna().sum()) for c in ["lon", "lat", "t"]}
            invalid = {"invalid_lon": int((~trk.lon.between(-180, 180)).sum()),
                       "invalid_lat": int((~trk.lat.between(-90, 90)).sum()),
                       "zero_lonlat": int(((trk.lon == 0) & (trk.lat == 0)).sum()),
                       "nonincreasing_t": int((dt <= 0).sum()),
                       "nan_nodes": int(tr.nodes.isna().sum().sum()), "nan_arcs": int(tr.arcs.isna().sum().sum()),
                       "nan_route": int(np.isnan(tr.route.astype(float)).sum())}
            per_trip.append({"partition": part, "trip_id": tr.trip_id, "n_obs": len(trk),
                             "n_nodes": len(tr.nodes), "n_arcs": len(tr.arcs), "route_arcs": len(tr.route),
                             **{f"{k}_kb": v / 1024 for k, v in sizes.items()}, **nan_cols, **invalid,
                             "gaps_gt5s": int((dt > 5).sum()), "gaps_gt30s": int((dt > 30).sum()),
                             "pct_dt_1s": float(np.mean(np.isclose(dt, 1.0)) * 100),
                             "jumps_gt60mps": int((spd > 60).sum()),
                             "speed_median_mps": float(np.median(spd)),
                             "gps_route_median_m": float(np.median(dr)), "gps_route_p95_m": float(np.percentile(dr, 95)),
                             "lon_c": trk.lon.mean(), "lat_c": trk.lat.mean()})
            per_obs.append(pd.DataFrame({"partition": part, "trip_id": tr.trip_id,
                                         "dt": np.r_[np.nan, dt], "speed": np.r_[np.nan, spd], "d_route": dr}))
            print(f"{part:12s} {tr.trip_id} obs={len(trk):6d} arcs={len(tr.arcs):7d} med d_route={np.median(dr):5.1f} m")
    return pd.DataFrame(per_trip), pd.concat(per_obs, ignore_index=True)


# ------------------------------------------------------------------ figures
def fig_inventory(pt):
    inv = pt.groupby("partition").agg(trips=("trip_id", "count"), total_obs=("n_obs", "sum"),
                                      obs_min=("n_obs", "min"), obs_median=("n_obs", "median"),
                                      obs_max=("n_obs", "max"), nodes_median=("n_nodes", "median"),
                                      arcs_median=("n_arcs", "median"), route_arcs_total=("route_arcs", "sum"))
    inv.to_csv(OUT / "inventory.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.logspace(np.log10(pt.n_obs.min()), np.log10(pt.n_obs.max()), 25)
    for p in ["train", "student_test"]:
        ax.hist(pt.n_obs[pt.partition == p], bins=bins, alpha=0.7, color=PCOL[p], label=p)
    ax.set_xscale("log"); ax.set_xlabel("GPS observations per trip (log scale)"); ax.set_ylabel("trips")
    ax.set_title("Observations per trip  [train (80) + student_test (10)]"); ax.legend()
    save(fig, "fig01_obs_per_trip.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = ["track_kb", "nodes_kb", "arcs_kb", "route_kb"]
    data, labels, colors = [], [], []
    for c in cols:
        for p in ["train", "student_test"]:
            data.append(pt.loc[pt.partition == p, c]); labels.append(f"{c[:-3]}\n{p}"); colors.append(PCOL[p])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(c); b.set_alpha(0.6)
    ax.set_yscale("log"); ax.set_ylabel("file size (KB, log)")
    ax.set_title("Raw file sizes per trip  [train + student_test]")
    save(fig, "fig02_file_sizes.png")
    return inv


def fig_missing(pt):
    cols = ["nan_lon", "nan_lat", "nan_t", "invalid_lon", "invalid_lat", "zero_lonlat", "nonincreasing_t",
            "nan_nodes", "nan_arcs", "nan_route"]
    mv = pt.groupby("partition")[cols].sum().T
    mv.to_csv(OUT / "missing_values.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    im = ax.imshow(mv.to_numpy(), cmap="Reds", vmin=0, vmax=max(1, mv.to_numpy().max()), aspect="auto")
    ax.set_xticks(range(mv.shape[1]), mv.columns); ax.set_yticks(range(mv.shape[0]), mv.index)
    for i in range(mv.shape[0]):
        for j in range(mv.shape[1]):
            ax.text(j, i, int(mv.iat[i, j]), ha="center", va="center", fontsize=9)
    ax.grid(False)
    ax.set_title("Missing / invalid values per column (counts)  [train + student_test]")
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, "fig03_missing_values.png")
    return mv


def fig_time(pt, po):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for p in ["train", "student_test"]:
        dt = po.loc[(po.partition == p) & po.dt.notna(), "dt"].clip(upper=120)
        ax.hist(dt, bins=np.arange(0, 122, 1), alpha=0.65, color=PCOL[p], label=f"{p} (n={len(dt):,})", density=True)
    ax.set_yscale("log"); ax.set_xlabel("Δt between consecutive fixes (s, clipped at 120)")
    ax.set_ylabel("density (log)"); ax.legend()
    ax.set_title("Sampling interval distribution  [train + student_test]")
    save(fig, "fig04_dt_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    s = pt.sort_values("gaps_gt5s", ascending=False).reset_index(drop=True)
    c = [PCOL[p] for p in s.partition]
    axes[0].bar(range(len(s)), s.gaps_gt5s, color=c, label="Δt > 5 s")
    axes[0].bar(range(len(s)), s.gaps_gt30s, color="k", alpha=0.7, label="Δt > 30 s")
    axes[0].set_yscale("symlog"); axes[0].set_xlabel("trip (sorted)"); axes[0].set_ylabel("gaps per trip")
    axes[0].legend(); axes[0].set_title("Time gaps per trip  [train blue / student_test orange]")
    s = pt.sort_values("pct_dt_1s").reset_index(drop=True)
    axes[1].bar(range(len(s)), s.pct_dt_1s, color=[PCOL[p] for p in s.partition])
    axes[1].set_ylim(40, 100); axes[1].set_xlabel("trip (sorted)"); axes[1].set_ylabel("% of steps with Δt = 1 s")
    axes[1].set_title("Share of 1 Hz steps per trip  [train blue / student_test orange]")
    save(fig, "fig05_gaps_and_1s_share.png")


def fig_speed(pt, po):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for p in ["train", "student_test"]:
        v = po.loc[(po.partition == p) & po.speed.notna(), "speed"] * 3.6
        axes[0].hist(v.clip(upper=300), bins=150, alpha=0.65, color=PCOL[p], density=True, label=p)
    axes[0].axvline(216, color="r", ls="--", lw=1, label="216 km/h (60 m/s) jump threshold")
    axes[0].set_yscale("log"); axes[0].set_xlabel("implied speed between fixes (km/h, clipped 300)")
    axes[0].set_ylabel("density (log)"); axes[0].legend(fontsize=7)
    axes[0].set_title("Implied speed  [train + student_test]")
    s = pt[pt.jumps_gt60mps > 0].sort_values("jumps_gt60mps", ascending=False)
    axes[1].bar(s.trip_id.str[-2:], s.jumps_gt60mps, color=[PCOL[p] for p in s.partition])
    axes[1].set_yscale("log"); axes[1].set_xlabel("trip id (last 2 digits)"); axes[1].set_ylabel("steps > 60 m/s")
    axes[1].set_title(f"Implausible jumps: {len(s)} of {len(pt)} trips affected  [train + student_test]")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    save(fig, "fig06_speed_and_jumps.png")


def fig_noise(po):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    po = po[~po.trip_id.isin(TRUNCATED_ROUTE)]
    for p in ["train", "student_test"]:
        d = po.loc[po.partition == p, "d_route"]
        ax.hist(d.clip(upper=60), bins=np.arange(0, 61, 1), alpha=0.65, color=PCOL[p], density=True,
                label=f"{p}: median {d.median():.1f} m, p95 {d.quantile(.95):.1f} m, >60 m {np.mean(d > 60)*100:.1f}%")
    ax.set_xlabel("distance GPS fix → reference route (m, clipped at 60)"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.set_title("GPS noise vs reference route  [train excl. truncated-route trips 02/33/54 + student_test]", fontsize=8.5)
    save(fig, "fig07_gps_to_route_distance.png")


def fig_world(pt):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for p in ["train", "student_test"]:
        s = pt[pt.partition == p]
        ax.scatter(s.lon_c, s.lat_c, s=12 + s.n_obs / 150, c=PCOL[p], alpha=0.7, edgecolor="k", lw=0.3,
                   label=f"{p} ({len(s)} trips)")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal"); ax.legend(loc="lower left")
    ax.set_title("Trip centres worldwide (marker size ∝ #obs)  [train + student_test]")
    save(fig, "fig08_world_trip_centres.png")


def _load(part, tid):
    d = dict(PARTS)[part] / tid
    tr = load_trip(d, with_route=True)
    proj = make_projector(tr.track.lon.mean(), tr.track.lat.mean())
    return tr, proj, RoadNet(tr.nodes, tr.arcs, proj)


def fig_trajectories():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (part, tid) in zip(axes, TRAJ_TRIPS):
        tr, proj, _ = _load(part, tid)
        gx, gy = proj.transform(tr.track.lon.to_numpy(), tr.track.lat.to_numpy())
        t = tr.track.t.to_numpy() - tr.track.t.iloc[0]
        sc = ax.scatter(gx / 1000, gy / 1000, c=t / 60, s=2, cmap="viridis")
        ax.plot(gx / 1000, gy / 1000, lw=0.3, c="k", alpha=0.4)
        ax.set_aspect("equal"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
        ax.set_title(f"trip {tid}  [{part}]\n{len(gx)} fixes")
        fig.colorbar(sc, ax=ax, fraction=0.04, label="minutes")
    fig.suptitle("GPS trajectories in per-trip AEQD metres (colour = time)")
    save(fig, "fig09_trajectories.png")


def fig_networks():
    fig, axes = plt.subplots(1, len(NET_TRIPS), figsize=(5 * len(NET_TRIPS), 4.6))
    for ax, (part, tid) in zip(axes, NET_TRIPS):
        tr, proj, net = _load(part, tid)
        gx, gy = proj.transform(tr.track.lon.to_numpy(), tr.track.lat.to_numpy())
        pad = 400
        x0, x1, y0, y1 = gx.min() - pad, gx.max() + pad, gy.min() - pad, gy.max() + pad
        inside = ((np.maximum(net.ax, net.bx) > x0) & (np.minimum(net.ax, net.bx) < x1) &
                  (np.maximum(net.ay, net.by) > y0) & (np.minimum(net.ay, net.by) < y1))
        segs = np.stack([np.c_[net.ax, net.ay], np.c_[net.bx, net.by]], axis=1)[inside]
        ax.add_collection(LineCollection(segs, colors="#9a9a9a", linewidths=0.4))
        r = tr.route
        ax.add_collection(LineCollection(np.stack([np.c_[net.ax[r], net.ay[r]], np.c_[net.bx[r], net.by[r]]], 1),
                                         colors="#1a9e3f", linewidths=2.2, alpha=0.8, label="reference route"))
        ax.scatter(gx, gy, s=1.5, c="#1f6fd1", label="GPS fixes", zorder=3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"trip {tid}  [{part}]\n{net.n_arcs:,} arcs total, {inside.sum():,} shown")
        ax.legend(loc="upper right", fontsize=7, markerscale=4)
    fig.suptitle("Road network (grey) with GPS overlay")
    save(fig, "fig10_road_networks.png")


def fig_network_stats():
    inv = pd.read_csv(RESULTS / "stage1" / "trip_inventory.csv", dtype={"trip_id": str})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for p in ["train", "student_test"]:
        s = inv[inv.partition == p]
        axes[0].scatter(s.n_nodes, s.n_arcs, c=PCOL[p], s=14, label=p, alpha=0.8)
        axes[1].hist(s.n_weak_components, bins=np.logspace(0, np.log10(500), 20), color=PCOL[p], alpha=0.65, label=p)
        axes[2].hist(s.largest_wcc_frac * 100, bins=np.arange(65, 101, 1.5), color=PCOL[p], alpha=0.65, label=p)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("#nodes"); axes[0].set_ylabel("#arcs")
    axes[0].set_title("Network size per trip")
    axes[1].set_xscale("log"); axes[1].set_xlabel("#weakly connected components"); axes[1].set_ylabel("trips")
    axes[1].set_title("Fragmentation")
    axes[2].set_xlabel("% of nodes in largest weak component"); axes[2].set_ylabel("trips")
    axes[2].set_title("Largest-component fraction")
    for a in axes:
        a.legend(fontsize=7)
    fig.suptitle("Road-network size and connectivity  [train + student_test]  (from results/stage1/trip_inventory.csv)")
    save(fig, "fig11_network_size_connectivity.png")


def folium_maps():
    out = []
    for part, tid in MAP_TRIPS:
        tr, proj, net = _load(part, tid)
        gps = tr.track[["lat", "lon"]].to_numpy()
        segs = network_segments_near(tr, radius_m=150, net=net)
        ref = [net.arc_latlon(int(a)) for a in tr.route]
        html = MAPS / f"eda_{part}_{tid}.html"
        make_trip_map(gps, network_segments=segs, reference_coords=ref,
                      title=f"PathSnap EDA – trip {tid} [{part}]: GPS vs reference route", out_html=html)
        print("  wrote", html.name, f"({len(segs)} nearby road segments)")
        out.append(html)
    return out


def screenshot(htmls):
    """PNG of each Folium map for the report (headless Chrome via Playwright; skipped if unavailable)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  playwright not installed - open the HTML maps and screenshot them manually"); return
    import shutil
    exe = shutil.which("google-chrome") or shutil.which("chromium")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 820})
        for h in htmls:
            pg.goto(h.resolve().as_uri()); pg.wait_for_timeout(4000)
            if not pg.evaluate("typeof L !== 'undefined'"):      # Leaflet CDN unreachable -> map would be blank
                print("  Leaflet did not load (offline?) - skipped", h.with_suffix(".png").name); continue
            pg.screenshot(path=str(h.with_suffix(".png")))
            print("  wrote", h.with_suffix(".png").name)
        b.close()


def summary(pt, po, inv, mv):
    tr, te = pt[pt.partition == "train"], pt[pt.partition == "student_test"]
    trn_ok = po.loc[(po.partition == "train") & ~po.trip_id.isin(TRUNCATED_ROUTE), "d_route"]
    def q(p, col, f):
        return f(po.loc[(po.partition == p) & po[col].notna(), col])
    lines = [
        "# EDA summary (Report §2.2)",
        "",
        "Generated by `eda/02_eda.py`. Partitions: **train (80 trips)** and **student_test (10 trips)**; the evaluator",
        "partition is never read. Every number below comes from that script run.",
        "",
        "## Inventory  [train + student_test]",
        inv.to_markdown(floatfmt=".0f"),
        "",
        "## Missing / invalid values  [train + student_test]  (fig03)",
        mv.to_markdown(),
        "",
        "All counts are zero: no NaN, no out-of-range or (0,0) coordinates, timestamps strictly increasing.",
        "Preprocessing therefore needs no imputation; the real data issues are irregular sampling, jumps and imperfect routes.",
        "",
        "## Sampling  (fig04, fig05)",
        f"* Δt median: train {q('train','dt',np.median):.0f} s, student_test {q('student_test','dt',np.median):.0f} s; "
        f"share of 1 s steps per trip: train median {tr.pct_dt_1s.median():.1f} % (min {tr.pct_dt_1s.min():.1f} %), "
        f"student_test median {te.pct_dt_1s.median():.1f} % (min {te.pct_dt_1s.min():.1f} %).",
        f"* Gaps > 5 s: train {tr.gaps_gt5s.sum()}, student_test {te.gaps_gt5s.sum()}; gaps > 30 s: train {tr.gaps_gt30s.sum()}, "
        f"student_test {te.gaps_gt30s.sum()}.",
        "",
        "## Speed and jumps  (fig06)",
        f"* Median implied speed: train {q('train','speed',np.median)*3.6:.1f} km/h, student_test {q('student_test','speed',np.median)*3.6:.1f} km/h.",
        f"* Steps > 60 m/s (216 km/h): {int(pt.jumps_gt60mps.sum())} in {int((pt.jumps_gt60mps > 0).sum())} trips; worst: "
        + ", ".join(f"{r.trip_id} [{r.partition}] {r.jumps_gt60mps}" for r in pt.nlargest(3, 'jumps_gt60mps').itertuples()) + ".",
        "",
        "## GPS noise vs reference route  (fig07)",
        f"* All fixes pooled – train: median {q('train','d_route',np.median):.1f} m, p95 {q('train','d_route',lambda x: np.percentile(x,95)):.1f} m; "
        f"student_test: median {q('student_test','d_route',np.median):.1f} m, p95 {q('student_test','d_route',lambda x: np.percentile(x,95)):.1f} m.",
        f"* The pooled train p95 is dominated by the truncated-route trips {sorted(TRUNCATED_ROUTE)} (trip 54 alone has "
        f"{int(tr.loc[tr.trip_id=='00000054','n_obs'].iloc[0]):,} fixes, most beyond the end of its route). Excluding them: train median "
        f"{np.median(trn_ok):.1f} m, p95 {np.percentile(trn_ok,95):.1f} m (this is what fig07 shows).",
        f"* Per-trip p95, median over trips: train {tr.gps_route_p95_m.median():.1f} m, student_test {te.gps_route_p95_m.median():.1f} m "
        "→ basis for the candidate search radius and the HMM emission σ.",
        "",
        "## Geography and networks  (fig08–fig11)",
        f"* Trip centres span lon {pt.lon_c.min():.1f}…{pt.lon_c.max():.1f}, lat {pt.lat_c.min():.1f}…{pt.lat_c.max():.1f}, "
        "so distances use a per-trip azimuthal-equidistant projection.",
        f"* Network size: {pt.n_arcs.min():,} – {pt.n_arcs.max():,} arcs per trip (median {int(pt.n_arcs.median()):,}).",
        "",
        "## Folium maps  (results/maps/)",
        *[f"* `eda_{p}_{t}.html` (+ `.png`) – trip {t} [{p}]: GPS, nearby network (150 m), reference route, layer control, legend."
          for p, t in MAP_TRIPS],
        "",
        "## Figure index",
        *[f"* `{f.name}`" for f in sorted(OUT.glob("fig*.png"))],
    ]
    (OUT / "eda_summary.md").write_text("\n".join(lines) + "\n")
    print("  wrote eda_summary.md")


def main():
    pt, po = collect()
    pt.to_csv(OUT / "per_trip_eda.csv", index=False)
    inv = fig_inventory(pt)
    mv = fig_missing(pt)
    fig_time(pt, po)
    fig_speed(pt, po)
    fig_noise(po)
    fig_world(pt)
    fig_trajectories()
    fig_networks()
    fig_network_stats()
    htmls = folium_maps()
    screenshot(htmls)
    summary(pt, po, inv, mv)


if __name__ == "__main__":
    main()
