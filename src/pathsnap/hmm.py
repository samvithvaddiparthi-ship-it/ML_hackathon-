"""HMM + Viterbi route reconstruction on top of the candidate classifier.

Hidden states
    At observation i the hidden state is "the vehicle is on candidate arc c at along-arc position t" for every
    candidate (arc_id, t_along) generated for that observation (<= 12 per observation). So the state space changes
    from step to step and the candidate -> road-segment correspondence is kept explicitly.

Emission  log P(z_i | c) = w_e * log(p_clf(c | z_i) + 1e-6)
    p_clf is the selected classifier's probability that the candidate is the true arc (it already folds in distance,
    heading, neighbour support, ...). w_e trades the classifier against the road-network transition model.

Transition  log P(c_i | c_{i-1}) = -| d_route - d_gps | / beta      (Newson & Krumm 2009 style)
    d_gps   = straight-line distance between the two GPS fixes,
    d_route = distance driven along the DIRECTED network from (arc a, t_a) to (arc b, t_b):
              same arc: |t_b - t_a| * len(a)  (small backward moves = GPS noise),
              else    : (1 - t_a) len(a) + shortest_path(head(a) -> tail(b)) + t_b len(b).
    If b is not reachable from a within a speed-based cut-off the transition is impossible (-inf).

Gaps / disconnections
    * dt > GAP_S seconds (large GPS gap): the network constraint is dropped for that step (uniform transition)
      - the vehicle may have driven anywhere, so we only keep the emission evidence.
    * If NO candidate of step i is reachable from any candidate of step i-1 (disconnected network pieces,
      gross GPS jump), the chain is re-started at step i (new segment) instead of forcing an impossible path.
    Both events are counted and returned (n_gap_breaks, n_disconnect_breaks).

Decoding: standard Viterbi (max-product in log space) with back-pointers; then the decoded arc sequence is
expanded into a continuous route by inserting the shortest-path arcs between consecutive distinct arcs.
"""
import heapq
import numpy as np

DEFAULTS = {"beta": 10.0, "w_e": 1.0, "gap_s": 60.0, "vmax": 50.0}
_BUCKETS = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0)


class Graph:
    """Directed adjacency (CSR) of a RoadNet with a cached, cut-off Dijkstra."""

    def __init__(self, net):
        self.net = net
        order = np.argsort(net.u, kind="stable")
        self.arc_sorted = order
        self.v_sorted = net.v[order]
        self.len_sorted = net.length[order]
        self.indptr = np.r_[0, np.cumsum(np.bincount(net.u, minlength=len(net.x)))]
        self.cache = {}

    def sssp(self, src, cutoff):
        """Shortest distances (and incoming arc) from node src to nodes within cutoff metres."""
        b = next((x for x in _BUCKETS if x >= cutoff), _BUCKETS[-1])
        key = (int(src), b)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        dist = {int(src): 0.0}
        pred = {}
        pq = [(0.0, int(src))]
        ip, vs, ls, arcs = self.indptr, self.v_sorted, self.len_sorted, self.arc_sorted
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, np.inf) or d > b:
                continue
            for k in range(ip[n], ip[n + 1]):
                m = int(vs[k]); nd = d + ls[k]
                if nd <= b and nd < dist.get(m, np.inf):
                    dist[m] = nd; pred[m] = int(arcs[k])
                    heapq.heappush(pq, (nd, m))
        self.cache[key] = (dist, pred)
        return dist, pred

    def path_arcs(self, src, dst, cutoff=12800.0):
        """Arc ids of the shortest path src -> dst (empty if src == dst, None if unreachable)."""
        if src == dst:
            return []
        dist, pred = self.sssp(src, cutoff)
        if dst not in dist:
            return None
        out, n = [], dst
        while n != src:
            a = pred[n]; out.append(a); n = int(self.net.u[a])
        return out[::-1]


def _route_dist(g, a_prev, t_prev, a_cur, t_cur, cutoff):
    """Matrix of along-network distances between all candidate pairs (inf = unreachable)."""
    net = g.net
    L = net.length
    D = np.full((len(a_prev), len(a_cur)), np.inf)
    for j, (a, ta) in enumerate(zip(a_prev, t_prev)):
        dist, _ = g.sssp(net.v[a], cutoff)
        rest = (1.0 - ta) * L[a]
        for k, (b, tb) in enumerate(zip(a_cur, t_cur)):
            if a == b:
                D[j, k] = abs(tb - ta) * L[a]
            else:
                s = dist.get(int(net.u[b]))
                if s is not None:
                    D[j, k] = rest + s + tb * L[b]
    return D


def viterbi(cands, prob, t, gx, gy, net, params=None, graph=None):
    """cands: DataFrame with columns obs, arc, t_along (rows grouped by obs, every obs 0..n-1 present).
    prob: classifier probabilities aligned with cands rows. Returns dict with per-observation arcs + diagnostics."""
    P = {**DEFAULTS, **(params or {})}
    g = graph or Graph(net)
    obs = cands["obs"].to_numpy(); arcs = cands["arc"].to_numpy(); ta = cands["t_along"].to_numpy()
    n = int(obs.max()) + 1
    starts = np.searchsorted(obs, np.arange(n + 1))
    logE = P["w_e"] * np.log(np.clip(prob, 0, 1) + 1e-6)

    s0, e0 = starts[0], starts[1]
    delta = logE[s0:e0].copy()
    backs = [None]
    n_gap = n_disc = 0
    for i in range(1, n):
        s, e = starts[i], starts[i + 1]
        ps, pe = starts[i - 1], starts[i]
        dt = t[i] - t[i - 1]
        dgps = float(np.hypot(gx[i] - gx[i - 1], gy[i] - gy[i - 1]))
        if dt > P["gap_s"]:
            j = int(np.argmax(delta)); backs.append(np.full(e - s, j)); n_gap += 1
            delta = delta[j] + logE[s:e]
        else:
            cutoff = min(P["vmax"] * max(dt, 1.0) + 50.0, 3.0 * dgps + 150.0)
            D = _route_dist(g, arcs[ps:pe], ta[ps:pe], arcs[s:e], ta[s:e], cutoff)
            logT = -np.abs(D - dgps) / P["beta"]
            tot = delta[:, None] + logT
            bp = np.argmax(tot, axis=0)
            best = tot[bp, np.arange(e - s)]
            if not np.isfinite(best).any():                     # disconnected: restart the chain
                j = int(np.argmax(delta)); bp = np.full(e - s, j); best = np.full(e - s, delta[j]); n_disc += 1
            backs.append(bp)
            delta = best + logE[s:e]
        delta = delta - np.max(delta)                           # keep numbers bounded
    # back-tracking
    k = int(np.argmax(delta))
    choice = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        choice[i] = starts[i] + k
        if i > 0:
            k = int(backs[i][k])
    return {"pred_arc": arcs[choice], "row": choice, "n_gap_breaks": n_gap, "n_disconnect_breaks": n_disc, "graph": g}


def expand_route(arc_seq, g):
    """Collapse repeats and fill the arcs driven between consecutive distinct arcs with the shortest path."""
    seq = [int(a) for a in arc_seq if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if not seq:
        return []
    out = [seq[0]]
    net = g.net
    for b in seq[1:]:
        a = out[-1]
        mid = g.path_arcs(int(net.v[a]), int(net.u[b]), cutoff=3200.0)
        if mid:
            out.extend(mid)
        out.append(b)
    return [a for i, a in enumerate(out) if i == 0 or a != out[i - 1]]
