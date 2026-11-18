"""Raw-file readers. Every trip folder <id>/ holds <id>.track/.nodes/.arcs/.route (headerless, whitespace-separated)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame        # lon, lat, t
    nodes: pd.DataFrame        # lon, lat   (index = node_id)
    arcs: pd.DataFrame         # u, v       (index = arc_id)
    route: np.ndarray | None   # arc_id sequence (None if not supplied)


def _read(path: Path, names, dtype=None):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(trip_dir: Path, with_route: bool = True) -> Trip:
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"], float)
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"], float)
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], np.int64)
    route = None
    if with_route:
        r = _read(trip_dir / f"{tid}.route", ["arc"], np.int64)
        route = None if r is None else r["arc"].to_numpy()
    return Trip(tid, track, nodes, arcs, route)


def list_trips(partition_dir: Path) -> list[Path]:
    return sorted(p for p in Path(partition_dir).iterdir() if p.is_dir())
