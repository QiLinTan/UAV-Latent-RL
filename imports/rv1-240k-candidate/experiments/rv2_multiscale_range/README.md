# rv2_multiscale_range — engineering candidate v1

This directory is independent of the running rv1 trainer. No formal training entry point is provided.

* Canonical input: the frozen rv1 4096-ray 32x128 yaw-horizontal scan, 8m, 15Hz. Quantize first with `8/65534` and construct both online/Replay images with exactly `q.astype(float32)/65534`.
* Radial bins: `[0,1.5)`, `[1.5,4)`, `[4,8]` m. Each has 5 channels: hit in bin, local hit depth (zero unless hit in bin), visible free fraction, blocked before bin, scanned. `[0,.08)` stays unknown. Fractions describe ray intervals, not inter-ray volumes.
* All branches consume the full angular grid; no raw angular pooling is enabled. Near strides 1/2/2, Mid/Far 2/2/2. Channels 12/20/30. Learned feature pooling to 4x16, projections 24/24/16. Encoder 147430 parameters versus frozen single-scale 145472.
* Optional deterministic bypass: horizontal 8 sectors centered at 0,45,...315deg; vertical `[-30,-10)`, `[-10,10)`, `[10,30]` deg. Per sector nearest valid near hit /1.5 and scanned-direction fraction, interleaved. 48D; Actor input 130D, otherwise 82D. No controller/action correction.
* Encoders `single`, `multiscale`, `joint_bands`, `single_large`. `single` exactly wraps the frozen rv1 Encoder. Capacity options 24D/32D/96D bypass are separate contracts; 48D is the primary candidate.
* Actor/Critic have independent encoders; twin Q heads share the critic encoder. TD3 update equations are inherited unchanged from the frozen Direct Velocity implementation. Current adapter is CPU-based; CUDA consistency remains unverified on this host.
* Raw Replay stores State18, uint16 ranges, packed hit/scanned masks, projected local actions and terminal metadata. Learned latent and bypass are reconstructed. Tensor ZIP archives avoid the NumPy/pickle2 full-buffer issue. Existing destination files are rejected.
* A checkpoint is an engineering network/optimizer/RNG checkpoint, not an autonomous Full-physics training-resume bundle. Formal training still needs its own versioned environment/course/episode capture and acceptance.

Tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 .wsl2/drones-cpu/bin/python -B -m pytest \
 experiments/rv2_multiscale_range/tests -q -p no:cacheprovider
```

Geometry/cost audit (new output directory required):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 .wsl2/drones-cpu/bin/python -B -m experiments.rv2_multiscale_range.audit \
 experiments/rv2_multiscale_range/outputs/engineering_NEW
```

The audit uses fixed original Step3 train/validation maps, excludes sealed test, and reads only the predeclared rv1 40k actor encoder for contextual frozen-feature probes. Four TD3 updates per variant are computational smoke tests, not RL navigation training. Results do not establish navigation gains.
