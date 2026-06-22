"""Guardrail for the unfinished AvoidBench control-environment integration."""


def main():
    raise SystemExit(
        "AvoidBench's avoidbridge module is a Unity rendering/collision bridge, "
        "not an RL environment: it has no action step, reward, termination, or reset API. "
        "Run `python -m scripts.probe_avoidbench` to validate images first. "
        "Training requires a separate ROS control/dynamics adapter before TD3 can be connected."
    )


if __name__ == "__main__":
    main()
