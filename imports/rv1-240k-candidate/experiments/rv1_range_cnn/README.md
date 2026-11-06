# rv1_range_cnn

Independent Step 3 experiment. The sibling `direct_velocity_full_v1` is imported
read-only; no old weights, Replay or experiment files are written. The frozen
Gate A acceptance_v2 controller, 18D State, velocity projection, 120Hz reward and
termination functions are reused. Dense sensing runs at the 15Hz boundary only.

## Sensor API

`LidarConfig(height, width, horizontal_fov_deg, vertical_min_deg,
vertical_max_deg, azimuth_start_deg, max_range_m, origin_offset_m,
origin_height_m, batch_size)` defines an elevation-major H x W grid. Horizontal
360-degree sampling excludes its repeated endpoint; vertical sampling includes
both endpoints. `from_resolution` accepts exact divisible degree resolutions.
Angles, not ray count alone, define the observation. Positive elevation points up.

World frame is the project's right-handed XYZ, meters, seconds, xyzw quaternion.
The virtual scanner is yaw-horizontal, not body-tilted: yaw rotates the local
directions, roll/pitch do not. The centre is position + [0,0,0.02]m; each ray
starts 0.08m away from that centre to avoid self intersection. The blind interval
inside that offset remains unknown. Range includes the offset and is measured
from the scanner centre. `point = origin_world + range * direction_world`.

`Lidar.scan(client_id=..., position=..., quaternion_xyzw=..., timestamp_s=...,
cache_key=..., scanned_mask=..., self_ids=..., ground_ids=..., obstacle_ids=...)`
returns an immutable `Scan`. The timestamp is simulation time, not wall time.
Hit-body IDs/types and raw PyBullet hit coordinates are diagnostic fields only.
The policy receives exactly State18 and range/hit/scanned arrays, with no map IDs,
geometry, QA route, collision list or hit-type channel.

* Scanned + hit: measured first surface; preceding interval is ray-visible free;
  the interval beyond it is unknown, including additional obstacles.
* Scanned + no hit: range=max_range; free only along this ray up to that limit.
* Unscanned: range=NaN, scanned=false, hit=false; all space on this ray is unknown.
* Gaps between rays have no occupancy guarantee. A mask is not an occupancy map.

Partial masks exist to test the data contract; no adaptive/active scan policy is
implemented. Full scanning is the experiment default. `Scan.visibility` queries
only along a specified ray. Point clouds include only actual first hits from the
same scan and are used for QA, not policy inputs.

## Storage and future comparisons

`RangeCodec` stores uint16 round(range / (max_range/65534)); 65535 marks unscanned.
Hit and scanned masks are separately bit-packed. No pooling/interpolation occurs.
No-hit and hit-at-limit remain distinguishable by the hit mask. A future Replay
must store sensor inputs, not a stale learned Latent. A transition stores both
observations explicitly; chain deduplication is an estimated future option only.

The default sensor is 32 x 128, 360 x 60 degrees, 8m. Density comparisons retain
these FOV limits and range. The separate 120-degree vertical experiment changes
coverage and must not be reported as a pure resolution improvement. 32 x 128 and
16 x 256 trade vertical against horizontal samples; neither implies a universal
detection guarantee.

Challenge seeds and all variants are declared in `benchmark.py`. All variants are
retained, including policy failures. New validation is deliberately exposed by
this study; the new test split is reserved for later method decisions. Old
Forest validation/test files are untouched. Offline QA can access complete map
truth, while Actor inputs cannot. A waypoint route label does not implement a
multi-candidate RL planner or prove all possible homotopy classes are enumerated.

## Execution order

1. Run `tests/test_geometry.py`.
2. Run `python -B -m experiments.rv1_range_cnn.audit_geometry OUTPUT`.
3. Only after PASS_GEOMETRY_COST, test the CNN/TD3 stack and run the bounded smoke.
4. Evaluate the frozen dv0 checkpoint on the full predeclared challenge validation.
5. Generate figures/report and verify protected assets. No formal long training
   and no multiscale/gating/GRU/feasibility/active-scanning/planner modules here.

CPU simulation and CPU/GPU inference timings must be reported separately. This
checkout currently uses a CPU-only Torch runtime; GPU time cannot be inferred.
