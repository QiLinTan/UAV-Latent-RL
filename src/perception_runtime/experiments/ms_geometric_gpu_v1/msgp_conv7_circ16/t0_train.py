"""Train the original RV1 CNN with the frozen E11 obstacle course."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from experiments.ms_geometric_mixed.mixed_course import describe, make_mixed_env
from experiments.rv1_range_cnn import train_formal


def main():
    run = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve()
    historical = Path("/opt/uav-training-20261103/runs/directional_mixed_seed43_40k_20261119a")
    protocol = describe()
    actual = json.loads(json.dumps(protocol))
    if actual != json.loads((historical / "mixed_protocol.json").read_text()):
        raise RuntimeError("E11 mixed protocol changed")
    cfg = train_formal.load_config()
    original_cfg = json.loads((historical / "original/metadata.json").read_text())["config"]
    for key in ("seed", "td3", "reward_scale", "reward_kwargs", "control_spec", "curriculum"):
        if json.loads(json.dumps(cfg[key])) != original_cfg[key]:
            raise RuntimeError(f"E11 task setting changed: {key}")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def make_env(pool):
        env = make_mixed_env(pool)
        reset = env.reset

        def record_reset(**kwargs):
            observation, info = reset(**kwargs)
            log = run / "episode_starts.jsonl"
            train_formal.log(log, dict(
                episode=env._pool_draw_count - 1, stage=info["density_stage"],
                map_pool_index=info["map_pool_index"], geometry_sha256=info["geometry_sha256"],
                start_level=info["start_curriculum_episode_level"],
                start_source=info["start_curriculum_source"],
            ))
            return observation, info

        env.reset = record_reset
        return env

    train_formal.make_course_env = make_env
    train_formal.main()


if __name__ == "__main__":
    main()
