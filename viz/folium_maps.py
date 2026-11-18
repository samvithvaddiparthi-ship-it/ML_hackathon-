"""Reusable Folium map builder for PathSnap (EDA maps, HMM comparison maps, report figures).

Layer colours (fixed so every map in the report reads the same way):
    GPS observations     blue dots
    road network         thin grey lines
    reference route      green
    classifier-only      orange dashed
    HMM / predicted      red
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np
from branca.element import MacroElement, Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.geo import RoadNet, make_projector  # noqa: E402

COLORS = {"gps": "#1f6fd1", "network": "#8a8a8a", "reference": "#1a9e3f",
          "classifier": "#f28c00", "predicted": "#d62728"}


def _legend(title: str, items: list[tuple[str, str, str]]) -> MacroElement:
    """items = [(label, colour, kind)] with kind in {'dot', 'line', 'dash'}."""
    rows = []
    for label, col, kind in items:
        if kind == "dot":
            sym = f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{col};margin-right:6px"></span>'
        else:
            style = "dashed" if kind == "dash" else "solid"
            sym = f'<span style="display:inline-block;width:22px;border-top:3px {style} {col};margin-right:6px;vertical-align:middle"></span>'
        rows.append(f"<div>{sym}{label}</div>")
    head = f"<div style='font-weight:600;margin-bottom:4px'>{title}</div>" if title else ""
    html = f"""{{% macro html(this, kwargs) %}}
<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;
            border:1px solid #bbb;border-radius:6px;font:13px/1.5 sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.2)">
{head}{''.join(rows)}</div>{{% endmacro %}}"""
    el = MacroElement()
    el._template = Template(html)
    return el


def _polyline_layer(segments, name, color, weight, dash=None, opacity=0.9, show=True):
    fg = folium.FeatureGroup(name=name, show=show)
    for seg in segments:
        folium.PolyLine(seg, color=color, weight=weight, opacity=opacity, dash_array=dash).add_to(fg)
    return fg


def make_trip_map(gps_latlon, network_segments=None, reference_coords=None, predicted_coords=None,
                  classifier_only_coords=None, title="", out_html=None) -> folium.Map:
    """Build a layered map of one trip.

    gps_latlon             [[lat, lon], ...]            raw observations
    network_segments       [[[lat,lon],[lat,lon]], ...]  road arcs near the track (see network_segments_near)
    reference_coords       same format, ground-truth route arcs (offline/report use only - never sent to the UI)
    predicted_coords       same format, final (HMM) route
    classifier_only_coords same format, route from the classifier alone
    """
    gps = np.asarray(gps_latlon, dtype=float)
    m = folium.Map(location=gps.mean(axis=0).tolist(), zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    legend = []

    if network_segments:
        _polyline_layer(network_segments, "Road network (nearby)", COLORS["network"], 1.5, opacity=0.6).add_to(m)
        legend.append(("Road network", COLORS["network"], "line"))
    if reference_coords:
        _polyline_layer(reference_coords, "Reference route", COLORS["reference"], 6, opacity=0.75).add_to(m)
        legend.append(("Reference route", COLORS["reference"], "line"))
    if classifier_only_coords:
        _polyline_layer(classifier_only_coords, "Classifier-only route", COLORS["classifier"], 4, dash="6 6").add_to(m)
        legend.append(("Classifier only", COLORS["classifier"], "dash"))
    if predicted_coords:
        _polyline_layer(predicted_coords, "Predicted route (HMM)", COLORS["predicted"], 3).add_to(m)
        legend.append(("Predicted (HMM)", COLORS["predicted"], "line"))

    fg = folium.FeatureGroup(name=f"GPS observations ({len(gps)})")
    step = max(1, len(gps) // 4000)                      # keep huge trips light in the browser
    for lat, lon in gps[::step]:
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0).add_to(fg)
    fg.add_to(m)
    legend.insert(0, ("GPS observations", COLORS["gps"], "dot"))
    folium.Marker(gps[0].tolist(), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(gps[-1].tolist(), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    if title:
        m.get_root().html.add_child(folium.Element(
            f"<div style='position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;"
            f"border-radius:6px;font:600 14px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.2)'>{title}</div>"))
    m.get_root().add_child(_legend("", legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([gps.min(axis=0).tolist(), gps.max(axis=0).tolist()], padding=(20, 20))
    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_html))
    return m


def network_segments_near(trip, gps=None, radius_m: float = 150.0, net: RoadNet | None = None):
    """Lat/lon 2-point segments of every arc with a sample point within radius_m of a GPS observation.
    `trip` is a pathsnap.io.Trip; gps defaults to its track. Two-way roads are drawn once.
    If you pass a prebuilt `net`, it must use make_projector(mean lon, mean lat of the GPS) - as
    pipeline.predict_trip does - so both live in the same metric frame."""
    trk = trip.track
    lon, lat = (trk.lon.to_numpy(), trk.lat.to_numpy()) if gps is None else (np.asarray(gps)[:, 1], np.asarray(gps)[:, 0])
    if net is None:
        net = RoadNet(trip.nodes, trip.arcs, make_projector(lon.mean(), lat.mean()))
    proj = make_projector(lon.mean(), lat.mean())
    gx, gy = proj.transform(lon, lat)
    hits = net.tree.query_ball_point(np.c_[gx, gy], r=radius_m)
    idx = np.unique(np.concatenate([np.asarray(h, dtype=int) for h in hits])) if len(hits) else np.empty(0, int)
    arcs = np.unique(net.sample_arc[idx])
    seen, segs = set(), []
    for a in arcs:
        key = frozenset((int(net.u[a]), int(net.v[a])))
        if key in seen:
            continue
        seen.add(key)
        segs.append(net.arc_latlon(int(a)))
    return segs
