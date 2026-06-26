from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.probe_avoidbench_collision_ownership import (
    DEFAULT_CONTAINER_CONFIG,
    DirectBridgeProbe,
    jsonable,
    write_csv,
    write_json,
    write_jsonl,
)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_collision(bridge: DirectBridgeProbe) -> bool | None:
    try:
        return bool(bridge.adapter.collision())
    except Exception:
        return None


def safe_scene_changed(bridge: DirectBridgeProbe) -> bool | None:
    try:
        return bool(bridge.adapter.bridge.ifSceneChanged())
    except Exception:
        return None


def make_state(
    bridge: DirectBridgeProbe,
    *,
    position: list[float],
    timestamp: float,
):
    return bridge.adapter.create_state(
        position=position,
        orientation=(0.0, 0.0, 0.0, 1.0),
        velocity=[0.0, 0.0, 0.0],
        timestamp=timestamp,
    )


def run_case(
    args,
    *,
    scenario: str,
    position: list[float],
    case_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bridge = DirectBridgeProbe(
        args.config,
        spawn_obstacles=False,
        mission_end=tuple(args.mission_end),
        mission_radius=args.mission_radius,
        mission_seed=args.mission_seed,
    )
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    update_success_count = 0
    spawn_return = None
    scene_changed = None
    time_to_scene_changed = None
    first_true_elapsed = None

    def record(
        phase: str,
        *,
        update_return: bool | None = None,
        error: str = "",
    ) -> None:
        nonlocal first_true_elapsed
        collision = safe_collision(bridge)
        elapsed = time.monotonic() - started
        if collision is True and first_true_elapsed is None:
            first_true_elapsed = elapsed
        rows.append(
            {
                "case_index": int(case_index),
                "scenario": scenario,
                "phase": phase,
                "elapsed_s": elapsed,
                "position": list(position),
                "velocity": [0.0, 0.0, 0.0],
                "height": float(position[2]),
                "update_return": update_return,
                "update_success_count": int(update_success_count),
                "collision": collision,
                "scene_changed": safe_scene_changed(bridge),
                "spawn_return": spawn_return,
                "error": error,
            }
        )

    record("bridge_created_before_update")

    try:
        state = make_state(bridge, position=position, timestamp=0.0)
        update_return = bool(bridge.adapter.update_unity(state))
        update_success_count += int(update_return)
        record("after_first_update", update_return=update_return)
    except Exception as exc:
        record("after_first_update_error", error=f"{type(exc).__name__}: {exc}")

    if scenario == "spawn":
        bridge.adapter.configure_mission(
            start_point=(*position, 0.0),
            end_point=tuple(args.mission_end),
            trials=1,
            radius=args.mission_radius,
            seed=args.mission_seed,
            opacity=0.5,
            pointcloud_file="pointcloud-bridge-lifecycle",
        )
        try:
            spawn_return = bool(bridge.adapter.bridge.spawnObstacles())
        except Exception as exc:
            record("spawn_error", error=f"{type(exc).__name__}: {exc}")
        record("after_spawn_obstacles")

    deadline = time.monotonic() + args.update_seconds
    update_index = 0
    while time.monotonic() < deadline:
        try:
            if scenario == "spawn":
                bridge.adapter.bridge.SpawnNewObs()
            state = make_state(
                bridge,
                position=position,
                timestamp=float(update_index + 1) * args.sample_period,
            )
            update_return = bool(bridge.adapter.update_unity(state))
            update_success_count += int(update_return)
            scene_changed = safe_scene_changed(bridge)
            if scene_changed and time_to_scene_changed is None:
                time_to_scene_changed = time.monotonic() - started
            record("update_loop", update_return=update_return)
        except Exception as exc:
            record("update_loop_error", error=f"{type(exc).__name__}: {exc}")
            break
        update_index += 1
        time.sleep(args.sample_period)

    collision_values = [row["collision"] for row in rows]
    summary = {
        "case_index": int(case_index),
        "scenario": scenario,
        "position": list(position),
        "sample_count": len(rows),
        "before_update_collision": rows[0]["collision"] if rows else None,
        "after_first_update_collision": next(
            (row["collision"] for row in rows if row["phase"] == "after_first_update"),
            None,
        ),
        "collision_true_count": int(sum(value is True for value in collision_values)),
        "collision_false_count": int(sum(value is False for value in collision_values)),
        "first_true_elapsed_s": first_true_elapsed,
        "update_success_count": int(update_success_count),
        "spawn_return": spawn_return,
        "scene_changed": scene_changed,
        "time_to_scene_changed": time_to_scene_changed,
    }
    return summary, rows


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# AvoidBench Bridge Lifecycle Report",
        "",
        f"Run: `{payload['summary']['output_dir']}`",
        "",
        "## Result",
        "",
    ]
    for case in payload["summary"]["cases"]:
        lines.append(
            "- scenario={scenario} position={position} before_update={before} "
            "after_first_update={after} true_count={true_count} "
            "scene_changed={scene_changed} spawn_return={spawn_return}".format(
                scenario=case["scenario"],
                position=case["position"],
                before=case["before_update_collision"],
                after=case["after_first_update_collision"],
                true_count=case["collision_true_count"],
                scene_changed=case["scene_changed"],
                spawn_return=case["spawn_return"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If `before_update_collision` is false but `after_first_update_collision` is true, "
            "the collision is coming from Unity output during `updateUnity()`, not from a "
            "local default true value.",
            "",
            "Stage 1 remains blocked until lifecycle probes show collision-free reset and "
            "stable static observation.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trace AvoidBench direct bridge collision across lifecycle phases."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONTAINER_CONFIG)
    parser.add_argument(
        "--position",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "Z"),
        help=(
            "Position to test. Repeatable only with --allow-shared-bridge-state. "
            "Defaults to (0,0,5), the high-altitude discriminator."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("no-spawn", "spawn"),
        action="append",
        help="Lifecycle scenario to run. Repeatable. Defaults to both scenarios.",
    )
    parser.add_argument("--update-seconds", type=float, default=5.0)
    parser.add_argument("--sample-period", type=float, default=0.1)
    parser.add_argument("--mission-end", type=float, nargs=3, default=(0.0, 15.0, 2.0))
    parser.add_argument("--mission-radius", type=float, default=2.0)
    parser.add_argument("--mission-seed", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=Path("runs/avoidbench_bridge_lifecycle"))
    parser.add_argument(
        "--allow-shared-bridge-state",
        action="store_true",
        help=(
            "Allow multiple cases in one Python process. Use only to demonstrate "
            "UnityBridge::getInstance() state contamination; clean cases require "
            "one process per case."
        ),
    )
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"--config does not exist: {args.config}")
    if args.update_seconds < 0.0:
        parser.error("--update-seconds must be non-negative.")
    if args.sample_period <= 0.0:
        parser.error("--sample-period must be positive.")

    positions = args.position or [[0.0, 0.0, 5.0]]
    scenarios = args.scenario or ["no-spawn", "spawn"]
    if len(positions) * len(scenarios) > 1 and not args.allow_shared_bridge_state:
        parser.error(
            "UnityBridge is a process-local singleton, so multiple lifecycle cases "
            "in one process can reuse collision state. Run one --position and one "
            "--scenario per invocation, or pass --allow-shared-bridge-state only "
            "when intentionally measuring contamination."
        )
    output_dir = args.output_root / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    case_index = 0
    for scenario in scenarios:
        for position in positions:
            case_summary, rows = run_case(
                args,
                scenario=scenario,
                position=[float(value) for value in position],
                case_index=case_index,
            )
            case_summaries.append(case_summary)
            all_rows.extend(rows)
            case_index += 1

    summary = {
        "mode": "bridge-lifecycle",
        "output_dir": str(output_dir),
        "config": str(args.config),
        "update_seconds": float(args.update_seconds),
        "sample_period": float(args.sample_period),
        "case_count": len(case_summaries),
        "cases": case_summaries,
        "any_after_first_update_false": any(
            case["after_first_update_collision"] is False for case in case_summaries
        ),
        "all_after_first_update_true": all(
            case["after_first_update_collision"] is True for case in case_summaries
        )
        if case_summaries
        else None,
    }
    payload = {
        "args": vars(args),
        "summary": summary,
        "samples": all_rows,
    }
    write_json(output_dir / "summary.json", payload)
    write_jsonl(output_dir / "samples.jsonl", all_rows)
    write_csv(output_dir / "samples.csv", all_rows)
    write_report(output_dir / "report.md", payload)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
