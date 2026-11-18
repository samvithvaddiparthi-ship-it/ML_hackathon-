"""Geometry helpers: per-trip metric projection + road-network spatial index.

Why AEQD per trip: trips are spread over the whole world, so a single UTM zone is wrong; an
azimuthal-equidistant projection centred on the trip keeps distance error far below GPS noise
over a <100 km area, and lets us use fast planar maths in metres.
"""
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree


def make_projector(lon0: float, lat0: float) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m", always_xy=True)


class RoadNet:
    """Road network of one trip, in metres. Arcs are directed; arc_id = row index of .arcs."""

    def __init__(self, nodes, arcs, proj: Transformer, sample_step: float = 10.0):
        self.x, self.y = proj.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
        self.lon, self.lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
        self.u, self.v = arcs.u.to_numpy(), arcs.v.to_numpy()
        self.ax, self.ay = self.x[self.u], self.y[self.u]
        self.bx, self.by = self.x[self.v], self.y[self.v]
        self.dx, self.dy = self.bx - self.ax, self.by - self.ay
        self.length = np.hypot(self.dx, self.dy)
        self.heading = np.arctan2(self.dx, self.dy)          # radians, 0 = north, clockwise positive
        self.n_arcs = len(self.u)
        self.step = sample_step
        # densify every arc into points <= sample_step apart so a KD-tree can find long arcs too
        n = np.maximum(np.ceil(self.length / sample_step).astype(int), 1) + 1
        arc_rep = np.repeat(np.arange(self.n_arcs), n)
        start = np.repeat(np.cumsum(n) - n, n)
        frac = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
        sx = self.ax[arc_rep] + frac * self.dx[arc_rep]
        sy = self.ay[arc_rep] + frac * self.dy[arc_rep]
        self.sample_arc = arc_rep
        self.tree = cKDTree(np.c_[sx, sy])

    def project(self, px, py, arc_ids):
        """Distance and along-arc fraction t in [0,1] of points to given arcs (broadcast)."""
        L2 = np.maximum(self.length[arc_ids] ** 2, 1e-9)
        t = ((px - self.ax[arc_ids]) * self.dx[arc_ids] + (py - self.ay[arc_ids]) * self.dy[arc_ids]) / L2
        t = np.clip(t, 0.0, 1.0)
        qx = self.ax[arc_ids] + t * self.dx[arc_ids]
        qy = self.ay[arc_ids] + t * self.dy[arc_ids]
        return np.hypot(px - qx, py - qy), t

    def candidates(self, px, py, radius: float = 50.0, max_k: int = 12, min_k: int = 2):
        """For every GPS point return arrays (obs_idx, arc_id, dist, t) of candidate arcs within `radius`,
        keeping at most max_k closest. If fewer than min_k are found, the nearest arcs are taken anyway
        (so every observation always has candidates, even after large GPS errors)."""
        pts = np.c_[px, py]
        hits = self.tree.query_ball_point(pts, r=radius + self.step / 2)
        out_i, out_a, out_d, out_t = [], [], [], []
        for i, h in enumerate(hits):
            arcs = np.unique(self.sample_arc[h]) if len(h) else np.empty(0, int)
            if len(arcs) < min_k:
                _, nn = self.tree.query(pts[i], k=min(64, len(self.sample_arc)))
                arcs = np.unique(np.r_[arcs, self.sample_arc[np.atleast_1d(nn)]])
            d, t = self.project(px[i], py[i], arcs)
            keep = d <= radius
            if keep.sum() < min_k:
                keep = np.zeros_like(keep); keep[np.argsort(d)[:min_k]] = True
            arcs, d, t = arcs[keep], d[keep], t[keep]
            o = np.argsort(d)[:max_k]
            out_i.append(np.full(len(o), i)); out_a.append(arcs[o]); out_d.append(d[o]); out_t.append(t[o])
        return np.concatenate(out_i), np.concatenate(out_a), np.concatenate(out_d), np.concatenate(out_t)

    def arc_latlon(self, arc_id):
        return [[float(self.lat[self.u[arc_id]]), float(self.lon[self.u[arc_id]])],
                [float(self.lat[self.v[arc_id]]), float(self.lon[self.v[arc_id]])]]
