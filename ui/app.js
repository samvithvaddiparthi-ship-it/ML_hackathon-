/* PathSnap evaluator UI (Laptop 2). Contract: docs/INTERFACE.md §3.
 * GET /trips -> list, POST /predict {trip_id} -> GPS + predicted route + metrics.
 * The API never returns the reference route, and this UI never draws one. */
(() => {
  "use strict";
  const STORE_KEY = "pathsnap.apiUrl";
  const COLORS = { gps: "#1f6fd1", pred: "#d62728", clf: "#f28c00" };
  const $ = (id) => document.getElementById(id);

  const store = {
    get() { try { return localStorage.getItem(STORE_KEY); } catch (e) { return null; } },
    set(v) { try { localStorage.setItem(STORE_KEY, v); } catch (e) { /* storage unavailable */ } },
  };

  const params = new URLSearchParams(location.search);
  let apiUrl = (params.get("api") || store.get() || "http://localhost:8000").replace(/\/+$/, "");
  let selected = null;
  const done = new Set();

  // ---------------------------------------------------------------- map
  const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([20, 0], 2);
  const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  L.control.scale().addTo(map);
  const layers = { gps: L.layerGroup(), pred: L.layerGroup(), clf: L.layerGroup() };
  const ctrl = L.control.layers({ OpenStreetMap: osm }, {}, { collapsed: false }).addTo(map);
  let overlaysAdded = false, clfAdded = false;

  const legend = L.control({ position: "bottomleft" });
  legend.onAdd = () => {
    const d = L.DomUtil.create("div", "legend");
    d.innerHTML = `<div><i class="pt" style="background:${COLORS.gps}"></i>Raw GPS observations</div>
                   <div><i class="ln" style="border-color:${COLORS.pred}"></i>Predicted route</div>
                   <div id="legend-clf" class="hidden"><i class="dash" style="border-color:${COLORS.clf}"></i>Classifier only</div>`;
    return d;
  };
  legend.addTo(map);

  function drawResult(res) {
    // detach and refill so the draw order is always GPS < classifier < predicted (canvas draws in add order)
    Object.values(layers).forEach((g) => { g.clearLayers(); map.removeLayer(g); });
    const gps = res.gps || [];
    const step = Math.max(1, Math.floor(gps.length / 5000));   // keep very long trips responsive
    for (let i = 0; i < gps.length; i += step) {
      L.circleMarker(gps[i], { radius: 2.4, color: COLORS.gps, weight: 0, fillOpacity: 0.55 })
        .bindTooltip(`obs ${i} · t=${res.t ? res.t[i] : "?"} s · arc ${res.pred_arc ? res.pred_arc[i] : "?"}`)
        .addTo(layers.gps);
    }
    if (gps.length) {
      L.circleMarker(gps[0], { radius: 7, color: "#fff", weight: 2, fillColor: "#1a9e3f", fillOpacity: 1 })
        .bindTooltip("Start").addTo(layers.gps);
      L.circleMarker(gps[gps.length - 1], { radius: 7, color: "#fff", weight: 2, fillColor: "#222", fillOpacity: 1 })
        .bindTooltip("End").addTo(layers.gps);
    }
    (res.route_coords || []).forEach((seg, k) =>
      L.polyline(seg, { color: COLORS.pred, weight: 3, opacity: 0.9 })
        .bindTooltip(`predicted arc ${res.route_arcs ? res.route_arcs[k] : k}`).addTo(layers.pred));
    const clf = res.route_coords_classifier_only;              // optional key added by later pipeline versions
    if (Array.isArray(clf) && clf.length) {
      clf.forEach((seg) => L.polyline(seg, { color: COLORS.clf, weight: 3, dashArray: "6 6", opacity: 0.9 }).addTo(layers.clf));
    }
    $("legend-clf").classList.toggle("hidden", !(Array.isArray(clf) && clf.length));

    if (!overlaysAdded) {
      ctrl.addOverlay(layers.gps, "GPS observations");
      ctrl.addOverlay(layers.pred, "Predicted route");
      overlaysAdded = true;
    }
    const hasClf = layers.clf.getLayers().length > 0;
    if (hasClf && !clfAdded) { ctrl.addOverlay(layers.clf, "Classifier-only route"); clfAdded = true; }
    // add groups empty, in order, then fill: route lines always sit on top of the GPS dots
    const fill = { clf: layers.clf.getLayers(), pred: layers.pred.getLayers(), gps: layers.gps.getLayers() };
    ["gps", "clf", "pred"].forEach((k) => { layers[k].clearLayers(); layers[k].addTo(map); fill[k].forEach((l) => layers[k].addLayer(l)); });
    if (gps.length) map.fitBounds(L.latLngBounds(gps), { padding: [24, 24] });
  }

  // ---------------------------------------------------------------- helpers
  function banner(msg, kind = "error") {
    const b = $("banner");
    if (!msg) { b.classList.add("hidden"); return; }
    b.textContent = msg;
    b.className = "banner" + (kind === "info" ? " info" : "");
  }
  function setStatus(state, text) {
    const s = $("apiStatus");
    s.className = "status " + state;
    s.querySelector(".txt").textContent = text;
  }
  async function api(path, opts = {}, timeoutMs = 120000) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await fetch(apiUrl + path, { ...opts, signal: ctl.signal,
        headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail ? `${r.status}: ${body.detail}` : `HTTP ${r.status}`);
      return body;
    } catch (e) {
      if (e.name === "AbortError") throw new Error(`request timed out after ${timeoutMs / 1000} s`);
      if (e instanceof TypeError) throw new Error(`cannot reach ${apiUrl} – is the API running and the firewall open?`);
      throw e;
    } finally { clearTimeout(timer); }
  }
  const pct = (x) => (x === null || x === undefined ? null : (100 * x).toFixed(1) + "%");
  function setMetric(id, v, naText = "N/A") {
    const el = $(id);
    if (v === null || v === undefined) { el.textContent = naText; el.classList.add("na"); }
    else { el.textContent = v; el.classList.remove("na"); }
  }

  // ---------------------------------------------------------------- trips
  function renderTrips(trips) {
    const list = $("tripList");
    list.innerHTML = "";
    $("tripCount").textContent = `(${trips.length})`;
    if (!trips.length) { list.innerHTML = '<p class="muted">The API found no trip folders.</p>'; return; }
    trips.forEach((t, i) => {
      const el = document.createElement("div");
      el.className = "trip";
      el.dataset.id = t.trip_id;
      el.innerHTML = `<span><span class="id">${t.trip_id}</span>${done.has(t.trip_id) ? '<span class="done">✓ run</span>' : ""}</span>
                      <span class="meta">${t.n_obs.toLocaleString()} obs</span>`;
      el.title = `Trip ${i + 1} of ${trips.length}`;
      el.onclick = () => select(t.trip_id);
      el.ondblclick = () => { select(t.trip_id); run(); };
      list.appendChild(el);
    });
    if (selected) select(selected);
  }
  function select(id) {
    selected = id;
    document.querySelectorAll(".trip").forEach((e) => e.classList.toggle("selected", e.dataset.id === id));
    $("runBtn").disabled = false;
    $("runBtn").textContent = `Run map-matching on ${id}`;
  }

  async function connect() {
    apiUrl = $("apiUrl").value.trim().replace(/\/+$/, "") || "http://localhost:8000";
    if (!/^https?:\/\//.test(apiUrl)) apiUrl = "http://" + apiUrl;
    $("apiUrl").value = apiUrl;
    store.set(apiUrl);
    setStatus("busy", "connecting…");
    banner(null);
    try {
      const h = await api("/health", {}, 8000);
      setStatus("ok", `connected · ${h.model?.name || "model"} v${h.model?.version || "?"}`);
      const { trips } = await api("/trips", {}, 30000);
      renderTrips(trips);
    } catch (e) {
      setStatus("err", "unreachable");
      banner(`API error: ${e.message}`);
      $("tripList").innerHTML = '<p class="muted">Could not load trips.</p>';
      $("runBtn").disabled = true;
    }
  }

  async function run() {
    if (!selected) return;
    const id = selected;
    $("runBtn").disabled = true;
    $("spinner").classList.remove("hidden");
    banner(null);
    const t0 = performance.now();
    try {
      const useHmm = $("useHmm").checked;
      const res = await api("/predict", { method: "POST", body: JSON.stringify({ trip_id: id, use_hmm: useHmm }) });
      const total = performance.now() - t0;
      drawResult(res);
      const m = res.metrics || {};
      setMetric("m-top1", pct(m.top1_accuracy), "N/A – not yet available");
      setMetric("m-f1", m.macro_f1 == null ? null : m.macro_f1.toFixed(3), "N/A – not yet available");
      setMetric("m-prec", pct(m.route_precision));
      setMetric("m-rec", pct(m.route_recall));
      setMetric("m-rf1", m.route_f1 == null ? null : m.route_f1.toFixed(3));
      setMetric("m-cont", pct(m.route_continuity));
      setMetric("m-nobs", (res.n_obs ?? 0).toLocaleString());
      setMetric("m-lat", `${Math.round(res.server_latency_ms ?? 0)} / ${Math.round(total)} ms`);
      setMetric("m-model", res.model ? `${res.model.name} v${res.model.version}` : null);
      // optional keys added by the HMM pipeline - every one may be absent
      const hmm = res.hmm && typeof res.hmm === "object" ? res.hmm : null;
      const num = (v) => (typeof v === "number" ? v : null);
      if (!useHmm) setMetric("m-hmm", null, "off (classifier only)");
      else if (hmm && (num(hmm.n_gap_breaks) !== null || num(hmm.n_disconnect_breaks) !== null))
        setMetric("m-hmm", `${num(hmm.n_gap_breaks) ?? "–"} / ${num(hmm.n_disconnect_breaks) ?? "–"}`);
      else setMetric("m-hmm", null, "N/A – not reported by this pipeline");
      const tm = res.timing_ms && typeof res.timing_ms === "object" ? res.timing_ms : {};
      const parts = ["features", "classifier", "hmm"].filter((k) => num(tm[k]) !== null).map((k) => `${k} ${Math.round(tm[k])}`);
      const other = Object.entries(tm).filter(([k, v]) => !["features", "classifier", "hmm"].includes(k) && num(v) !== null)
        .map(([k, v]) => `${k} ${Math.round(v)}`);
      setMetric("m-timing", parts.concat(other).join(" · ") || null, "N/A");
      done.add(id);
      const el = document.querySelector(`.trip[data-id="${id}"] span`);
      if (el && !el.querySelector(".done")) el.insertAdjacentHTML("beforeend", '<span class="done">✓ run</span>');
    } catch (e) {
      banner(`Prediction for trip ${id} failed: ${e.message}`);
    } finally {
      $("spinner").classList.add("hidden");
      $("runBtn").disabled = false;
    }
  }

  // ---------------------------------------------------------------- wire up
  $("apiUrl").value = apiUrl;
  $("connectBtn").onclick = connect;
  $("apiUrl").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  $("runBtn").onclick = run;
  try { const v = localStorage.getItem("pathsnap.useHmm"); if (v !== null) $("useHmm").checked = v === "1"; } catch (e) { /* ignore */ }
  $("useHmm").onchange = () => { try { localStorage.setItem("pathsnap.useHmm", $("useHmm").checked ? "1" : "0"); } catch (e) { /* ignore */ } };
  if (location.protocol === "https:" && apiUrl.startsWith("http:")) {
    banner("This page is served over HTTPS; browsers block calls to an http:// API. Serve ui/ with python -m http.server.", "info");
  }
  connect();
})();
