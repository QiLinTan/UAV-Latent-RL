"""Portable map contracts for the lightweight PyBullet map atlas.

External sources use incompatible formats (SDF worlds, PLY point clouds and
mesh assets).  ``MapSpec`` is the small common contract used by this project
after conversion.  It is intentionally JSON-only so an experiment can record
the exact map, source and split without adding a YAML dependency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-three vector.")
    return tuple(float(component) for component in array)


def _positive_vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    vector = _vector3(value, name=name)
    if min(vector) <= 0.0:
        raise ValueError(f"{name} must contain only positive values.")
    return vector


@dataclass(frozen=True)
class StaticObstacleSpec:
    """One collision object in a converted map.

    ``cylinder`` and ``box`` are fully validated and usable immediately.
    ``mesh`` is the landing format for converted SubT/FlightBench assets; its
    route clearance cannot be certified from metadata alone, so the caller
    must provide a checked route or a converted occupancy representation.
    """

    kind: str
    position: tuple[float, float, float]
    radius: float | None = None
    height: float | None = None
    half_extents: tuple[float, float, float] | None = None
    mesh_path: str | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: tuple[float, float, float, float] = (0.36, 0.36, 0.40, 1.0)
    name: str = "obstacle"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, root: Path) -> "StaticObstacleSpec":
        kind = str(data.get("kind", "")).strip().lower()
        position = _vector3(data.get("position"), name="obstacle.position")
        color_values = np.asarray(data.get("color", (0.36, 0.36, 0.40, 1.0)), dtype=np.float32).reshape(-1)
        if color_values.shape != (4,) or not np.all(np.isfinite(color_values)):
            raise ValueError("obstacle.color must be a finite RGBA vector.")
        color = tuple(float(component) for component in color_values)
        name = str(data.get("name", kind or "obstacle"))

        if kind == "cylinder":
            radius = float(data.get("radius", 0.0))
            height = float(data.get("height", 0.0))
            if radius <= 0.0 or height <= 0.0:
                raise ValueError("cylinder obstacles require positive radius and height.")
            return cls(kind=kind, position=position, radius=radius, height=height, color=color, name=name)
        if kind == "box":
            half_extents = _positive_vector3(data.get("half_extents"), name="box.half_extents")
            return cls(kind=kind, position=position, half_extents=half_extents, color=color, name=name)
        if kind == "mesh":
            raw_path = data.get("mesh_path")
            if not raw_path:
                raise ValueError("mesh obstacles require mesh_path.")
            mesh_path = Path(str(raw_path))
            if not mesh_path.is_absolute():
                mesh_path = (root / mesh_path).resolve()
            return cls(
                kind=kind,
                position=position,
                mesh_path=str(mesh_path),
                mesh_scale=_positive_vector3(data.get("mesh_scale", (1.0, 1.0, 1.0)), name="mesh.mesh_scale"),
                color=color,
                name=name,
            )
        raise ValueError(f"Unsupported map obstacle kind: {kind!r}.")


@dataclass(frozen=True)
class MapValidationReport:
    map_id: str
    endpoints_valid: bool
    reachable: bool | None
    message: str


@dataclass(frozen=True)
class MapSpec:
    """A versioned, source-attributed navigation map.

    ``environment_overrides`` only applies to procedural forest maps.  Static
    map entries instead describe their collision primitives or converted mesh.
    """

    map_id: str
    source: str
    scenario: str
    split: str
    start_pos: tuple[float, float, float]
    target_pos: tuple[float, float, float]
    xy_half_extent: float
    obstacle_sensor_range: float
    obstacles: tuple[StaticObstacleSpec, ...] = ()
    environment_overrides: Mapping[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    license: str | None = None
    raw_asset_status: str = "local"
    notes: str = ""
    xy_half_extents: tuple[float, float] | None = None
    episode_len_sec: float | None = None
    navigation_z_bounds: tuple[float, float] | None = None
    box_collision_mode: str = "primitive"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, root: Path) -> "MapSpec":
        map_id = str(data.get("map_id", "")).strip()
        if not map_id:
            raise ValueError("map_id is required.")
        rectangular = data.get("xy_half_extents")
        if rectangular is not None:
            rectangular = np.asarray(rectangular, dtype=float)
            if rectangular.shape != (2,) or not np.isfinite(rectangular).all() or np.any(rectangular <= 0):
                raise ValueError("xy_half_extents must contain two finite positive values")
            rectangular = tuple(float(v) for v in rectangular)
        xy_half_extent = float(data.get("xy_half_extent", max(rectangular) if rectangular else 0.0))
        sensor_range = float(data.get("obstacle_sensor_range", 0.0))
        if xy_half_extent <= 0.0 or sensor_range <= 0.0:
            raise ValueError("xy_half_extent and obstacle_sensor_range must be positive.")
        obstacles = tuple(
            StaticObstacleSpec.from_dict(item, root=root)
            for item in data.get("obstacles", ())
        )
        overrides = dict(data.get("environment_overrides", {}))
        episode_len_sec = data.get("episode_len_sec")
        if episode_len_sec is not None:
            episode_len_sec = float(episode_len_sec)
            if not np.isfinite(episode_len_sec) or episode_len_sec <= 0:
                raise ValueError("episode_len_sec must be finite and positive")
        box_mode = data.get("box_collision_mode", "primitive")
        if box_mode not in ("primitive", "triangles"):
            raise ValueError("box_collision_mode must be primitive or triangles")
        z_bounds = data.get("navigation_z_bounds")
        if z_bounds is not None:
            z_bounds = np.asarray(z_bounds, dtype=float)
            if z_bounds.shape != (2,) or not np.isfinite(z_bounds).all() or not 0 <= z_bounds[0] < z_bounds[1]:
                raise ValueError("navigation_z_bounds must be two ordered finite nonnegative values")
            z_bounds = tuple(float(v) for v in z_bounds)
        return cls(
            map_id=map_id,
            source=str(data.get("source", "unknown")),
            scenario=str(data.get("scenario", "unknown")),
            split=str(data.get("split", "unspecified")),
            start_pos=_vector3(data.get("start_pos"), name="start_pos"),
            target_pos=_vector3(data.get("target_pos"), name="target_pos"),
            xy_half_extent=xy_half_extent,
            obstacle_sensor_range=sensor_range,
            obstacles=obstacles,
            environment_overrides=overrides,
            source_url=data.get("source_url"),
            license=data.get("license"),
            raw_asset_status=str(data.get("raw_asset_status", "local")),
            notes=str(data.get("notes", "")),
            xy_half_extents=rectangular,
            episode_len_sec=episode_len_sec,
            navigation_z_bounds=z_bounds,
            box_collision_mode=box_mode,
        )

    @property
    def uses_uncertified_mesh(self) -> bool:
        return any(obstacle.kind == "mesh" for obstacle in self.obstacles)

    def endpoint_inside_bounds(self, point: np.ndarray) -> bool:
        return bool(np.all(np.abs(point[:2]) <= self.bounds_xy))

    @property
    def bounds_xy(self) -> tuple[float, float]:
        return self.xy_half_extents or (self.xy_half_extent, self.xy_half_extent)

    def point_is_occupied(self, point: np.ndarray, *, drone_radius: float) -> bool:
        point = np.asarray(point, dtype=np.float32)
        for obstacle in self.obstacles:
            if obstacle.kind == "mesh":
                continue
            if obstacle.kind == "cylinder":
                assert obstacle.radius is not None and obstacle.height is not None
                low_z = obstacle.position[2] - obstacle.height / 2.0
                high_z = obstacle.position[2] + obstacle.height / 2.0
                if low_z - drone_radius <= point[2] <= high_z + drone_radius:
                    delta = point[:2] - np.asarray(obstacle.position[:2], dtype=np.float32)
                    if float(np.linalg.norm(delta)) <= obstacle.radius + drone_radius:
                        return True
            elif obstacle.kind == "box":
                assert obstacle.half_extents is not None
                center = np.asarray(obstacle.position, dtype=np.float32)
                extents = np.asarray(obstacle.half_extents, dtype=np.float32) + drone_radius
                if bool(np.all(np.abs(point - center) <= extents)):
                    return True
        return False

    def validate_route(
        self,
        *,
        drone_radius: float = 0.06,
        resolution: float = 0.10,
    ) -> MapValidationReport:
        """Validate endpoints and 2-D connectivity for primitive maps.

        The test is conservative: the vehicle radius inflates all obstacles.
        Mesh-only maps return ``reachable=None`` because collision geometry
        must first be converted to a voxel/primitive representation.
        """

        if self.navigation_z_bounds is not None:
            from .navigation3d import find_route_3d
            return find_route_3d(self, vehicle_radius=drone_radius, resolution=resolution)[0]

        start = np.asarray(self.start_pos, dtype=np.float32)
        target = np.asarray(self.target_pos, dtype=np.float32)
        endpoints_valid = (
            self.endpoint_inside_bounds(start)
            and self.endpoint_inside_bounds(target)
            and not self.point_is_occupied(start, drone_radius=drone_radius)
            and not self.point_is_occupied(target, drone_radius=drone_radius)
        )
        if not endpoints_valid:
            return MapValidationReport(self.map_id, False, False, "start or target is invalid/occupied")
        if self.uses_uncertified_mesh:
            return MapValidationReport(
                self.map_id,
                True,
                None,
                "mesh collision is loadable, but route validation requires a converted occupancy map",
            )

        half_x, half_y = self.bounds_xy
        xs = np.arange(-half_x, half_x + resolution * 0.5, resolution)
        ys = np.arange(-half_y, half_y + resolution * 0.5, resolution)

        def to_index(point: np.ndarray) -> tuple[int, int]:
            ix = int(np.clip(np.rint((point[0] + half_x) / resolution), 0, len(xs) - 1))
            iy = int(np.clip(np.rint((point[1] + half_y) / resolution), 0, len(ys) - 1))
            return ix, iy

        start_index = to_index(start)
        target_index = to_index(target)
        queue: deque[tuple[int, int]] = deque([start_index])
        visited = {start_index}
        while queue:
            index = queue.popleft()
            if index == target_index:
                return MapValidationReport(self.map_id, True, True, "primitive route is reachable")
            for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_index = (index[0] + delta_x, index[1] + delta_y)
                if (
                    next_index in visited
                    or not 0 <= next_index[0] < len(xs)
                    or not 0 <= next_index[1] < len(ys)
                ):
                    continue
                probe = np.array((xs[next_index[0]], ys[next_index[1]], start[2]), dtype=np.float32)
                if self.point_is_occupied(probe, drone_radius=drone_radius):
                    continue
                visited.add(next_index)
                queue.append(next_index)
        return MapValidationReport(self.map_id, True, False, "primitive route is blocked")


class MapCatalog:
    """Read the project map registry and resolve immutable ``MapSpec`` values."""

    def __init__(self, catalog_path: str | Path):
        self.path = Path(catalog_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Map catalog does not exist: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        maps = payload.get("maps", ())
        self._maps = {
            str(item["map_id"]): MapSpec.from_dict(item, root=self.path.parent)
            for item in maps
        }
        if len(self._maps) != len(maps):
            raise ValueError(f"Map catalog contains duplicate map_id values: {self.path}")

    def get(self, map_id: str) -> MapSpec:
        try:
            return self._maps[map_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._maps))
            raise KeyError(f"Unknown map_id {map_id!r}; available: {available}") from exc

    def ids(self, *, split: str | None = None) -> tuple[str, ...]:
        return tuple(
            map_id
            for map_id, spec in sorted(self._maps.items())
            if split is None or spec.split == split
        )


def coerce_map_spec(value: MapSpec | Mapping[str, Any] | None, *, root: str | Path | None = None) -> MapSpec | None:
    if value is None or isinstance(value, MapSpec):
        return value
    if isinstance(value, Mapping):
        return MapSpec.from_dict(value, root=Path(root or ".").resolve())
    raise TypeError("map_spec must be a MapSpec, mapping, or None.")
