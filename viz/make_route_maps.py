"""Stage B8 - classifier-only vs HMM route comparison maps (Report §5 / §6).

    python viz/make_route_maps.py --trip-dir "<...>/student_test/00000052" --out results/maps/00000052_routes.html
    python viz/make_route_maps.py --all-student-test            # one map per student-test trip

Layers: GPS (blue), road network within 150 m (grey), reference route (green, read here for the REPORT only -
never sent to the UI), classifier-only route (orange dashed, `route_coords_classifier_only` - skipped if the
pipeline does not return it yet), HMM route (red, `route_coords`). A PNG screenshot is written next to each HTML
when Playwright + internet are available (Leaflet CDN / map tiles); otherwise only the HTML is written.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from config import STUDENT_TEST_DIR, RESULTS                 # noqa: E402
from pathsnap.io import load_trip                           # noqa: E402
from pathsnap.geo import make_projector, RoadNet            # noqa: E402
from pathsnap.pipeline import predict_trip                  # noqa: E402
from pathsnap.evaluate import score_trip                    # noqa: E402
from viz.folium_maps import make_trip_map, network_segments_near  # noqa: E402


def build_map(trip_dir: Path, out_html: Path, with_reference: bool = True) -> dict:
    pred = predict_trip(trip_dir, use_hmm=True)
    tr = load_trip(trip_dir, with_route=with_reference)
    net = RoadNet(tr.nodes, tr.arcs, make_projector(tr.track.lon.mean(), tr.track.lat.mean()))
    ref = [net.arc_latlon(int(a)) for a in tr.route] if with_reference and tr.route is not None else None
    clf = pred.get("route_coords_classifier_only")
    m = score_trip(pred, trip_dir) if with_reference else {}
    f1 = m.get("route_f1")
    title = (f"Trip {tr.trip_id} – {pred['model']['name']} v{pred['model']['version']}"
             + (f" – route F1 {f1:.2f}" if f1 is not None else ""))
    make_trip_map(pred["gps"], network_segments=network_segments_near(tr, net=net, radius_m=150),
                  reference_coords=ref, predicted_coords=pred["route_coords"],
                  classifier_only_coords=clf or None, title=title, out_html=out_html)
    return {"trip_id": tr.trip_id, "html": out_html, "has_classifier_only": bool(clf), "metrics": m}


def screenshot(htmls: list[Path]):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed - skipping PNG screenshots"); return
    exe = shutil.which("google-chrome") or shutil.which("chromium")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 820})
        for h in htmls:
            pg.goto(h.resolve().as_uri()); pg.wait_for_timeout(4000)
            if not pg.evaluate("typeof L !== 'undefined'"):
                print(f"  Leaflet did not load (offline?) - no PNG for {h.name}"); continue
            pg.screenshot(path=str(h.with_suffix(".png"))); print("  wrote", h.with_suffix(".png").name)
        b.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--trip-dir", type=Path, help="one trip folder (<id>/<id>.track ...)")
    g.add_argument("--all-student-test", action="store_true", help="one map per student-test trip")
    ap.add_argument("--out", type=Path, help="output HTML (with --trip-dir); default results/maps/<id>_routes.html")
    ap.add_argument("--no-reference", action="store_true", help="do not draw the reference route (e.g. evaluator trips)")
    ap.add_argument("--no-png", action="store_true", help="skip PNG screenshots")
    a = ap.parse_args()

    out_dir = RESULTS / "maps"
    dirs = sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()) if a.all_student_test else [a.trip_dir]
    done = []
    for d in dirs:
        out = a.out if (a.out and not a.all_student_test) else out_dir / f"{d.name}_routes.html"
        r = build_map(d, out, with_reference=not a.no_reference)
        m = r["metrics"]
        print(f"{r['trip_id']}: {out}  classifier-only layer: {'yes' if r['has_classifier_only'] else 'no (not returned yet)'}"
              + (f"  route_f1={m['route_f1']:.3f}" if m.get("route_f1") is not None else ""))
        done.append(out)
    if not a.no_png:
        screenshot(done)


if __name__ == "__main__":
    main()
