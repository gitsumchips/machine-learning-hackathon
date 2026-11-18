"""Read, inspect, validate, and clean the raw trip files.

Each trip stores an ordered GPS track plus a separate graph. Node and arc IDs
are the zero-based row positions in their respective files; route values refer
to arc IDs, not to GPS observation rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd


@dataclass
class TripData:
    """One trip's cleaned observations, graph, and optionally loaded labels."""

    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: Optional[pd.Series] = None


def iter_trip_directories(split_dir: str | Path) -> Iterator[Path]:
    """Yield trip directories in deterministic order."""
    directory = Path(split_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Trip split directory does not exist: {directory}")
    yield from sorted(path for path in directory.iterdir() if path.is_dir())


def _read_whitespace_table(path: Path, columns: list[str], dtype: object) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required trip file is missing: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)
    if frame.shape[1] != len(columns):
        raise ValueError(
            f"{path} has {frame.shape[1]} columns; expected {len(columns)} ({columns})."
        )
    frame.columns = columns
    return frame


def _track_path(trip_dir: Path, trip_id: str) -> Path:
    # The supplied description says .track3, while the provided files use .track.
    for suffix in (".track", ".track3"):
        candidate = trip_dir / f"{trip_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No .track or .track3 file found in {trip_dir}")


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid GPS rows, order by time, and retain the first duplicate time.

    ``source_row_id`` preserves the original row position for inspection. GPS
    samples are not joined to nodes/arcs by row number.
    """
    required = ["longitude", "latitude", "timestamp_s"]
    missing = set(required) - set(track.columns)
    if missing:
        raise ValueError(f"Track is missing columns: {sorted(missing)}")

    result = track.loc[:, required].copy()
    result.insert(0, "source_row_id", track.index.to_numpy(dtype=np.int64))
    result = result.replace([np.inf, -np.inf], np.nan)
    valid = (
        result[required].notna().all(axis=1)
        & result["longitude"].between(-180, 180)
        & result["latitude"].between(-90, 90)
    )
    result = result.loc[valid].sort_values("timestamp_s", kind="stable")
    result = result.drop_duplicates("timestamp_s", keep="first").reset_index(drop=True)
    return result


def load_trip(trip_dir: str | Path, *, include_route: bool = False) -> TripData:
    """Load and validate one trip.

    Route labels are opt-in so test labels cannot accidentally enter feature
    preparation. File row positions define graph IDs and are preserved as
    DataFrame indices named ``node_id`` and ``arc_id``.
    """
    directory = Path(trip_dir)
    trip_id = directory.name
    raw_track = _read_whitespace_table(
        _track_path(directory, trip_id),
        ["longitude", "latitude", "timestamp_s"],
        np.float64,
    )
    nodes = _read_whitespace_table(
        directory / f"{trip_id}.nodes", ["longitude", "latitude"], np.float64
    )
    arcs = _read_whitespace_table(
        directory / f"{trip_id}.arcs", ["from_node_id", "to_node_id"], np.int64
    )

    if not np.isfinite(nodes.to_numpy()).all():
        raise ValueError(f"{trip_id}: nodes contain non-finite coordinates")
    if not nodes["longitude"].between(-180, 180).all() or not nodes["latitude"].between(-90, 90).all():
        raise ValueError(f"{trip_id}: node coordinates are outside valid longitude/latitude ranges")
    if len(arcs) and (
        arcs[["from_node_id", "to_node_id"]].to_numpy().min() < 0
        or arcs[["from_node_id", "to_node_id"]].to_numpy().max() >= len(nodes)
    ):
        raise ValueError(f"{trip_id}: arc endpoint references a missing node")

    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    route = None
    if include_route:
        route_frame = _read_whitespace_table(
            directory / f"{trip_id}.route", ["arc_id"], np.int64
        )
        route_values = route_frame["arc_id"]
        if len(route_values) and (
            route_values.min() < 0 or route_values.max() >= len(arcs)
        ):
            raise ValueError(f"{trip_id}: route references a missing arc")
        route = route_values.rename("arc_id")

    return TripData(
        trip_id=trip_id,
        track=clean_track(raw_track),
        nodes=nodes,
        arcs=arcs,
        route=route,
    )


def inspect_trip(trip_dir: str | Path) -> dict[str, object]:
    """Return row counts, inferred types, and basic sampling diagnostics."""
    directory = Path(trip_dir)
    trip_id = directory.name
    track = _read_whitespace_table(
        _track_path(directory, trip_id),
        ["longitude", "latitude", "timestamp_s"],
        np.float64,
    )
    nodes = _read_whitespace_table(
        directory / f"{trip_id}.nodes", ["longitude", "latitude"], np.float64
    )
    arcs = _read_whitespace_table(
        directory / f"{trip_id}.arcs", ["from_node_id", "to_node_id"], np.int64
    )
    route = _read_whitespace_table(directory / f"{trip_id}.route", ["arc_id"], np.int64)
    intervals = track["timestamp_s"].diff().dropna()
    return {
        "trip_id": trip_id,
        "track_file": _track_path(directory, trip_id).suffix,
        "track_rows": len(track),
        "node_rows": len(nodes),
        "arc_rows": len(arcs),
        "route_rows": len(route),
        "track_dtypes": ",".join(track.dtypes.astype(str)),
        "nodes_dtypes": ",".join(nodes.dtypes.astype(str)),
        "arcs_dtypes": ",".join(arcs.dtypes.astype(str)),
        "route_dtypes": ",".join(route.dtypes.astype(str)),
        "one_second_interval_fraction": float((intervals == 1).mean()) if len(intervals) else np.nan,
    }


def audit_split(split_dir: str | Path) -> pd.DataFrame:
    """Inspect every trip in a split; useful for checking naming and variation."""
    return pd.DataFrame(inspect_trip(path) for path in iter_trip_directories(split_dir))