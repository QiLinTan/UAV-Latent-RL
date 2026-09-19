# Route Obstacle TensorBoard Comparison

Common comparison uses steps <= 480000.

## Eval at 480k or nearest previous

| metric | nolatent | latentonly_soft |
| --- | ---: | ---: |
| eval/return | 294.1 | 161.6 |
| eval/final_goal_distance | 1.111 | 3.584 |
| eval/collision_rate | 0.0000 | 0.0000 |
| eval/truncated_rate | 1.000 | 1.000 |
| eval/success_rate | 0.0000 | 0.0000 |
| eval/length | 270.0 | 179.0 |
| eval/mean_abs_z_error | 0.2617 | 0.0442 |
| eval/final_z | 0.1709 | 1.058 |
| eval/min_z | 0.1709 | 1.000 |
| eval/mean_z_vel | -0.3691 | 0.0387 |

## Eval last 10 mean within common range

| metric | nolatent | latentonly_soft |
| --- | ---: | ---: |
| eval/return | 219.3 | 263.1 |
| eval/final_goal_distance | 1.931 | 1.611 |
| eval/collision_rate | 0.1000 | 0.1000 |
| eval/truncated_rate | 0.9000 | 0.9000 |
| eval/success_rate | 0.0000 | 0.0000 |
| eval/length | 228.1 | 232.1 |
| eval/mean_abs_z_error | 0.2244 | 0.1241 |
| eval/final_z | 0.2430 | 0.5719 |
| eval/min_z | 0.2430 | 0.5661 |
| eval/mean_z_vel | -0.3975 | -0.2111 |

## Last 200 training episodes/env records within common range

| metric | nolatent | latentonly_soft |
| --- | ---: | ---: |
| episode/return | 267.2 | 279.8 |
| episode/avg_reward | 1.007 | 1.120 |
| episode/length | 264.7 | 247.8 |
| env/collision | 0.0150 | 0.0900 |
| env/goal_distance | 1.097 | 1.191 |
| env/z_error | 0.8704 | 0.6067 |
| env/min_tree_clearance | 1.255 | 1.171 |
| env/reward/progress_reward | 1.457 | 1.410 |
| env/reward/route_progress | 0.0468 | 0.0475 |
| env/reward/height_penalty | 0.6963 | 0.4853 |
| env/reward/route_lateral_offset | 0.2354 | 0.5666 |
| env/reward/proximity_penalty | 0.0381 | 0.2716 |
| env/reward/collision_penalty | 0.3750 | 2.250 |

## Train diagnostics summary

| metric | nolatent | latentonly_soft |
| --- | ---: | ---: |
| critic_loss | mean=0.0002, last=0.0003 | mean=0.0003, last=0.0004 |
| actor_loss | mean=-0.3041, last=-0.4481 | mean=-0.3606, last=-0.5227 |
| actor_sat_pct | mean=0.7424, last=0.7539 | mean=0.3994, last=0.5020 |
| critic_encoder_grad_scale | NA | mean=0.0415, last=0.0500 |
| latent_std_mean | NA | mean=0.5508, last=0.8293 |
| latent_abs_mean | NA | mean=0.9042, last=0.9982 |
| recon_loss | NA | mean=0.0043, last=0.0002 |
| dyn_loss | NA | mean=0.0067, last=0.0002 |
| q_gap_abs_mean | NA | mean=0.0065, last=0.0085 |

## Plots

- eval_comparison.png
- training_env_comparison.png
- diagnostics_comparison.png
