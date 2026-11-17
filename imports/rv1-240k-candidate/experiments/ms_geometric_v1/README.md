# Multiscale perception prototype

This is an experimental RV1 observation encoder, not a validated navigation model.
Coordinates x,y,z are yaw-horizontal. Near and far order is documented in multiscale.py and the original local acceptance record. The reference DDA and batch-vectorized DDA use the same contract and treat unobserved space as unknown.

Run with the RV1 project root on PYTHONPATH and its PyBullet environment:

    python test_multiscale.py
    python scene_checks.py
    python bench_wall.py

On the training WSL machine, initial tests passed: 25 random batches of 200 rays, axis cases and 4 PyBullet scenes matched the reference voxel grid. Fixed wall range, occupied, occlusion, and outside-FOV assertions passed. A repeated static wall benchmark (100 frames) measured scan-to-features P50 16.37 ms and P95 19.72 ms; this excludes policy inference and control, is not a loaded-system real-time guarantee, and tests only one scene. Thin obstacle, aperture preservation, rotations, dynamics, collision envelope, and independent ray traversal oracle need further validation. No TD3 training performed.
