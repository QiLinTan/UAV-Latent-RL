import argparse
import pathlib
import shlex
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_base_cmd() -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.change_td3",
        "--seed",
        "42",
        "--total_steps",
        "500000",
        "--gui",
        "false",
        "--eval_gui",
        "false",
        "--eval_stepsleep",
        "false",
    ]


def build_mode_cmd(mode: str) -> list[str]:
    cmd = build_base_cmd()

    if mode == "baseline":
        cmd += [
            "--use_latent",
            "true",
            "--latent_input_scale",
            "0.1",
        ]
    elif mode == "v1trust":
        cmd += [
            "--use_latent",
            "true",
            "--actor_updates_encoder",
            "false",
            "--latent_input_scale",
            "0.1",
            "--trust_alpha",
            "0.5",
            "--trust_beta",
            "0.5",
            "--trust_q_min",
            "0.05",
            "--trust_q_max",
            "1.0",
            "--trust_ema_momentum",
            "0.99",
            "--trust_warmup_steps",
            "10000",
        ]
    elif mode == "nolantent":
        cmd += [
            "--use_latent",
            "false",
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Quick launcher for change_td3 experiment modes.")
    parser.add_argument(
        "mode",
        choices=["baseline", "v1trust", "nolantent"],
        help="Experiment preset to launch.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--gui", choices=["true", "false"], default=None)
    parser.add_argument("--eval_gui", choices=["true", "false"], default=None)
    parser.add_argument("--extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = build_mode_cmd(args.mode)

    if args.seed is not None:
        cmd[cmd.index("--seed") + 1] = str(args.seed)
    if args.total_steps is not None:
        cmd[cmd.index("--total_steps") + 1] = str(args.total_steps)
    if args.gui is not None:
        cmd[cmd.index("--gui") + 1] = args.gui
    if args.eval_gui is not None:
        cmd[cmd.index("--eval_gui") + 1] = args.eval_gui

    if args.extra:
        extra = args.extra
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    print("Launching:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
