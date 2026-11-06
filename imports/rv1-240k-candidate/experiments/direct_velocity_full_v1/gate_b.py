"""The training entry requires this successful, source-bound Gate B report."""
import argparse,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
from envs.map_atlas.forest_pool import ForestPool
from .task_factory import EXP,ROOT,load_config,data_contract,source_hashes,gate_a_check,make_reward


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if EXP/'outputs' not in output.parents:raise ValueError('isolated output required')
    output.mkdir(parents=True,exist_ok=False)
    c=load_config();gate_a_check(c);pool=ForestPool(ROOT/c['forest_pool_path']);contract=data_contract(c,pool)
    command=[sys.executable,'-B','-m','pytest','-p','no:cacheprovider','-q',str(EXP/'tests'),
        '--junitxml='+str(output/'tests.xml')]
    start=time.monotonic()
    with (output/'tests.log').open('w') as f:
        result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
    tests=list(ET.parse(output/'tests.xml').getroot().iter('testcase')) if (output/'tests.xml').exists() else []
    failed=[t.attrib['name'] for t in tests if t.find('failure') is not None or t.find('error') is not None or t.find('skipped') is not None]
    expected=['test_state_schema_normalization_and_invariance','test_action_timing_new_controller_and_no_old_calls',
        'test_rays_real_cast_yaw_zero_and_rotated_hits','test_reward_done_exact_task_contract',
        'test_curriculum_matches_old_pool_rng_candidates_and_reset','test_script_and_random_smoke',
        'test_schema_rejects_old_and_same_dimension_different_frame','test_replay_mask_stratification_roundtrip_and_projected_td3',
        'test_restore_episode_retains_control_reward_rng_and_next_transition']
    missing=[x for x in expected if x not in [t.attrib['name'] for t in tests]]
    report=dict(passed=result.returncode==0 and not failed and not missing,contract=contract,
        total_tests=len(tests),failures=failed,missing_required=missing,elapsed_s=time.monotonic()-start,
        groups=dict(state=True,coordinates=True,timing=True,controller=True,reward_done=True,curriculum_reset=True,
            real_raycast=True,data_schema=True,script_random=True,checkpoint_restore=True),
        source_hashes=source_hashes(),effective_reward=vars(make_reward(c)),range_gate='96-ray dv0 only; no high-density LiDAR claims')
    if not report['passed']:report['groups']={k:'UNACCEPTED' for k in report['groups']}
    (output/'gate_b.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print((output/'tests.log').read_text()[-18000:]);print(json.dumps(dict(passed=report['passed'],tests=len(tests),failures=failed,output=str(output))))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
