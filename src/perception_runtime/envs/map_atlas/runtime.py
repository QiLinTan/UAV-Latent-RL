"""Create and inspect MapSpec collision objects in a PyBullet client."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p

from .spec import MapSpec, StaticObstacleSpec


class MapRuntime:
    def __init__(self, spec: MapSpec):
        self.spec = spec
        self.body_ids: list[int] = []

    def create_bodies(self, *, client_id: int) -> list[int]:
        self.body_ids = [self._create_body(obstacle, client_id=client_id, box_triangles=self.spec.box_collision_mode == "triangles") for obstacle in self.spec.obstacles]
        return list(self.body_ids)

    @staticmethod
    def _create_body(obstacle: StaticObstacleSpec, *, client_id: int, box_triangles: bool = False) -> int:
        if obstacle.kind == "cylinder":
            assert obstacle.radius is not None and obstacle.height is not None
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=obstacle.radius,
                height=obstacle.height,
                physicsClientId=client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=obstacle.radius,
                length=obstacle.height,
                rgbaColor=obstacle.color,
                physicsClientId=client_id,
            )
        elif obstacle.kind == "box":
            assert obstacle.half_extents is not None
            if box_triangles:
                # Static triangle faces avoid convex-ray corner approximations
                # on closely adjoining cave slabs. Visual and QA extents match.
                x, y, z = obstacle.half_extents
                vertices = [[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
                            [-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]]
                faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),
                         (1,2,6,5),(2,3,7,6),(3,0,4,7)]
                indices = [v for a,b,c,d in faces for v in (a,b,c,a,c,d)]
                collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices,
                    indices=indices, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=client_id)
            else:
                collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=obstacle.half_extents,
                    physicsClientId=client_id,
                )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=obstacle.half_extents,
                rgbaColor=obstacle.color,
                physicsClientId=client_id,
            )
        elif obstacle.kind == "mesh":
            assert obstacle.mesh_path is not None
            mesh_path = Path(obstacle.mesh_path)
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"Converted mesh for map obstacle {obstacle.name!r} is missing: {mesh_path}. "
                    "Place source data in data/map_atlas/raw and its converted OBJ/STL in "
                    "data/map_atlas/converted before selecting this map."
                )
            collision = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=str(mesh_path),
                meshScale=obstacle.mesh_scale,
                physicsClientId=client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(mesh_path),
                meshScale=obstacle.mesh_scale,
                rgbaColor=obstacle.color,
                physicsClientId=client_id,
            )
        else:  # Defensive guard for specs constructed without from_dict().
            raise ValueError(f"Unsupported obstacle kind: {obstacle.kind!r}")
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=obstacle.position,
            physicsClientId=client_id,
        )

    def nearest_clearance(self, *, position, drone_radius: float, drone_height: float) -> float:
        position = np.asarray(position, dtype=np.float32)
        if self.spec.navigation_z_bounds is not None:
            # New 3D scenes use a conservative enclosing sphere, including caps.
            from .navigation3d import signed_clearance
            radius = float(np.hypot(drone_radius, drone_height / 2.0))
            return float(signed_clearance(self.spec, position[None, :])[0] - radius)
        best = np.inf
        for obstacle in self.spec.obstacles:
            clearance = self._clearance_to_obstacle(
                obstacle,
                position=position,
                drone_radius=drone_radius,
                drone_height=drone_height,
            )
            best = min(best, clearance)
        return float(best)

    @staticmethod
    def _clearance_to_obstacle(
        obstacle: StaticObstacleSpec,
        *,
        position: np.ndarray,
        drone_radius: float,
        drone_height: float,
    ) -> float:
        if obstacle.kind == "mesh":
            # Mesh clearance needs a precomputed signed-distance/voxel field;
            # collision and range sensing remain active in PyBullet.
            return np.inf
        if obstacle.kind == "cylinder":
            assert obstacle.radius is not None and obstacle.height is not None
            low_z = obstacle.position[2] - obstacle.height / 2.0
            high_z = obstacle.position[2] + obstacle.height / 2.0
            if position[2] - drone_height / 2.0 > high_z or position[2] + drone_height / 2.0 < low_z:
                return np.inf
            return float(
                np.linalg.norm(position[:2] - np.asarray(obstacle.position[:2], dtype=np.float32))
                - obstacle.radius
                - drone_radius
            )
        if obstacle.kind == "box":
            assert obstacle.half_extents is not None
            center_xy = np.asarray(obstacle.position[:2], dtype=np.float32)
            half_xy = np.asarray(obstacle.half_extents[:2], dtype=np.float32)
            low_z = obstacle.position[2] - obstacle.half_extents[2]
            high_z = obstacle.position[2] + obstacle.half_extents[2]
            if position[2] - drone_height / 2.0 > high_z or position[2] + drone_height / 2.0 < low_z:
                return np.inf
            outside = np.maximum(np.abs(position[:2] - center_xy) - half_xy, 0.0)
            return float(np.linalg.norm(outside) - drone_radius)
        raise ValueError(f"Unsupported obstacle kind: {obstacle.kind!r}")
