"""Zoomed PNG of an existing route map (results/maps/<trip>_routes.html) centred on the densest cluster of
observations where the classifier-only arc differs from the HMM arc. Used for report §6.

    python viz/zoom_screenshot.py 00000077 [zoom]
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from config import STUDENT_TEST_DIR, RESULTS
from pathsnap.pipeline import predict_trip


def main(tid, zoom=17):
    pred = predict_trip(STUDENT_TEST_DIR / tid)
    diff = np.flatnonzero(np.array(pred["pred_arc"]) != np.array(pred["pred_arc_classifier_only"]))
    gps = np.array(pred["gps"])
    # densest 60-observation window of disagreements
    best = max(range(len(diff)), key=lambda k: np.sum(np.abs(diff - diff[k]) <= 30)) if len(diff) else 0
    lat, lon = gps[diff[best]] if len(diff) else gps.mean(0)
    html = RESULTS / "maps" / f"{tid}_routes.html"
    out = RESULTS / "maps" / f"{tid}_routes_zoom.png"
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1100, "height": 700})
        pg.goto(html.resolve().as_uri()); pg.wait_for_timeout(2500)
        pg.evaluate(f"""() => {{ for (const k in window) {{ const m = window[k];
            if (m && m instanceof L.Map) m.setView([{lat}, {lon}], {zoom}); }} }}""")
        pg.wait_for_timeout(3500); pg.screenshot(path=str(out)); b.close()
    print(f"wrote {out}  centre=({lat:.5f},{lon:.5f})  disagreements={len(diff)}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 17)
