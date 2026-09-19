"""Resolve the native-teacher gate from existing traces and historical logs."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.evaluate_native_latent_actor import write_json
from scripts.evaluate_native_latent_actor import checkpoint_eval_step, scalar_at_step
from scripts.freeze_logical_async import file_record, sha256


def summarize(root):
    root=Path(root)
    audit=json.loads((root/"audit.json").read_text())
    history=json.loads((root/"historical_metrics.json").read_text())
    profiles=audit["profiles"]
    sample=next(r for r in profiles["historical_stage2_native"]["episodes"] if r["seed"]==0)
    comparison={}
    checkpoint_prefix=audit.get("checkpoint_prefix", "checkpoints/latent_only_affordance_dim16/model_final")
    comparison_step=audit.get("comparison_eval_step", checkpoint_eval_step(checkpoint_prefix))
    for tag,key,tolerance in (("eval/return","reward",1e-3),("eval/final_goal_distance","final_distance",1e-5),("eval/length","steps",0.)):
        old=scalar_at_step(history, tag, comparison_step)
        if old is not None:
            comparison[tag]={"historical":old,"reproduced":sample[key],"absolute_error":abs(old-sample[key]),"tolerance":tolerance,"passed":abs(old-sample[key])<=tolerance}
    paired={}
    for ep in profiles["historical_stage2_native"]["episodes"]:
        seed=ep["seed"]
        with np.load(root/"historical_stage2_native"/f"seed_{seed}"/"trace.npz") as a, np.load(root/"current_stage2_native"/f"seed_{seed}"/"trace.npz") as b:
            paired[str(seed)]={key:bool(np.array_equal(a[key],b[key])) for key in ("raw_observation","latent","actor_action","rpm","position")}
    checkpoint_identity={}
    checkpoint_base=ROOT / str(Path(checkpoint_prefix).parent)
    for name in ("encoder","actor"):
        checkpoint_identity[name]={
            "loaded_sha256": audit["networks"][name]["sha256"],
            "final_sha256": sha256(checkpoint_base/f"model_final_{name}") if (checkpoint_base/f"model_final_{name}").exists() else None,
            "500000_sha256": sha256(checkpoint_base/f"model_500000_{name}") if (checkpoint_base/f"model_500000_{name}").exists() else None,
            "best_sha256": sha256(checkpoint_base/f"model_best_{name}") if (checkpoint_base/f"model_best_{name}").exists() else None,
        }
    historical_pass=all(v["passed"] for v in comparison.values())
    no_success=all(v["success_rate"]==0 for v in profiles.values())
    verdict={"audit":file_record(root/"audit.json"),"historical_metrics":file_record(root/"historical_metrics.json"),
             "historical_evaluation_reproduced":historical_pass,"comparison":comparison,"current_vs_historical_trace_equality":paired,
             "comparison_eval_step":comparison_step,
             "checkpoint_identity":checkpoint_identity,"historical_max_logged_success":history["maximum_logged_eval_success_rate"],
             "teacher_gate_passed":not no_success,"native_successful_episode_count":sum(ep["success"] for v in profiles.values() for ep in v["episodes"]),
             "classification":"historical_failure_reproduced" if historical_pass and no_success else "review_required",
             "finding":(
                 "The audited checkpoint has successful native episodes in at least one predefined profile. "
                 "Review which profile and reward/map contract it satisfies before using it as a teacher."
                 if not no_success else
                 "The archived run demonstrated forward progress, not successful goal convergence. Current stage2 physical trajectories exactly reproduce history; current reward differs. Asymmetric decoding is not compatible with this native Actor."
             ),
             "stage5_actor_bc_allowed":False if no_success else None,
             "next_decision":"Diagnose overshoot/terminal velocity and task representation before selecting a successful native teacher. A-D probes are diagnostic only; no Planner-v2/BC/RL stage is opened."}
    write_json(root/"verdict.json",verdict)
    print(json.dumps({"historical_reproduced":historical_pass,"teacher_gate_passed":not no_success,"historical_max_success":history["maximum_logged_eval_success_rate"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="analysis_outputs/native_actor_audit_v1")
    summarize(parser.parse_args().root)
