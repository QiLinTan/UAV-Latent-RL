import copy
import unittest

import numpy as np

from scripts.audit_spatial_terminal_consistency import comparison_summary
from scripts.diagnose_terminal_reward_rollouts import preset_args, reward_model


class SpatialTerminalConsistencyTest(unittest.TestCase):
    def test_noroute_preset_changes_one_learning_parameter(self):
        baseline = vars(preset_args())
        alternative = vars(preset_args(
            "upper_dslpid_recurrent_multitree_terminal_v2_noroute_h8_k1"))
        output_fields = {
            "log_dir", "ckpt_dir", "trajectory_output_dir",
            "collision_diagnostic_output_dir", "timeout_diagnostic_output_dir",
            "eval_terminal_coverage_output_dir",
        }
        differences = {key for key in baseline if baseline[key] != alternative[key]}
        self.assertEqual(differences-output_fields, {"route_progress_reward_weight"})
        self.assertEqual(alternative["route_progress_reward_weight"], 0.0)
        self.assertEqual(alternative["goal_progress_reward_weight"], 8.0)
        self.assertIsNone(alternative["load_checkpoint"])
        for key in output_fields:
            self.assertIn("terminal_v2_noroute_h8_k1", alternative[key])

    def test_removing_route_changes_only_running_route_reward(self):
        args = preset_args()
        alternative_args = copy.copy(args)
        alternative_args.route_progress_reward_weight = 0.0
        baseline, alternative = reward_model(args), reward_model(alternative_args)
        previous = np.array([0.2, 0.1, 1.5])
        state = np.zeros(16)
        state[:3] = [0.25, 0.1, 1.52]
        state[10:13] = [0.75, 0.0, 0.3]
        target = np.array([1., 0., 1.])
        for reason in ("running", "success", "collision", "timeout", "height_bound"):
            with self.subTest(reason=reason):
                kwargs = dict(state=state, prev_goal_dist=np.linalg.norm(target-previous),
                    prev_pos=previous, start_pos=[-1.,0.,1.], target_pos=target,
                    goal_tolerance=0.2, safe_distance=0.35, clearance=0.25,
                    collision=reason=="collision", terminal_reason=reason)
                rb, tb, _ = baseline.compute(**kwargs)
                ra, ta, _ = alternative.compute(**kwargs)
                expected = 12*tb["route_progress"] if reason=="running" else 0.0
                self.assertAlmostEqual(rb-ra, expected)
                for key in tb:
                    if key not in ("progress_reward", "running_shaping_reward"):
                        self.assertEqual(tb[key], ta[key], key)

    def test_ranking_reports_best_and_all_successes_separately(self):
        rows = [dict(case_id="a", behavior=b, outcome=o, G_baseline=g, G_no_route=g-1)
                for b,o,g in (("fast","success",5), ("slow","success",-2),
                              ("hover","timeout",0))]
        rows.append(dict(case_id="b", behavior="actor", outcome="timeout",
                         G_baseline=0, G_no_route=0))
        result = comparison_summary(rows)["baseline"]
        self.assertEqual(result["comparable_cases"], 1)
        self.assertEqual(result["best_success_better"], 1)
        self.assertEqual(result["success_failure_pair_count"], 2)
        self.assertEqual(result["success_failure_pair_wins"], 1)
        self.assertEqual(result["cases_without_sampled_success"], ["b"])


if __name__ == "__main__":
    unittest.main()
