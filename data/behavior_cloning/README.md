# Behavior-cloning datasets

- `legacy_projected_v1/`: quarantined metadata for the historical fixed-0.05,
  collective/differential-projected teacher path. Training loaders must reject it.
- `asymmetric_rpm_v2/`: versioned T3 datasets generated with a reset
  `DSLPIDControl` and the asymmetric RPM codec.

No training script may discover datasets recursively. A dataset directory must
be supplied explicitly and pass the metadata assertions in
`data.behavior_cloning_dataset.BehaviorCloningDataset`.
