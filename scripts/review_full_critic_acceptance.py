"""Read-only consolidation of complete deterministic/noise critic audits.

Concurrent resumptions can append identical rows. Preserve all originals,
deduplicate exact experiment keys, and reject any conflicting duplicate.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.audit_full_critic_frozen import AUDIT_STEPS, calibration, map_bootstrap, summarize
from scripts.run_full_critic_frozen import DEFAULT_ROOT, production_hashes
from scripts.run_full_ft_v2 import dump


def unique_rows(path, expected_count):
    snapshot = path.read_bytes()
    rows = [json.loads(line) for line in snapshot.splitlines()]
    unique = {}
    for row in rows:
        key = (row['step'], row['case_id'], row['candidate'], row['replica'])
        if key in unique and unique[key] != row:
            raise RuntimeError(f'Conflicting audit records: {path} {key}')
        unique[key] = row
    if len(unique) != expected_count:
        raise RuntimeError(f'Incomplete evidence: {path}: {len(unique)}/{expected_count} unique rows')
    return list(unique.values()), {'path': str(path), 'snapshot_sha256': hashlib.sha256(snapshot).hexdigest(),
        'raw_rows': len(rows), 'unique_rows': len(unique), 'exact_duplicate_rows': len(rows)-len(unique),
        'conflicting_duplicates': 0}


def noise_changes(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['case_id'], row['candidate'])].append(row)
    for group in grouped.values():
        assert {r['replica'] for r in group} == {0, 1, 2, 3} and len(group) == 4
    average = {(case, name): {**group[0], 'G': float(np.mean([r['G'] for r in group]))}
               for (case, name), group in grouped.items()}
    outputs = {}
    for candidate in ('q1_best_grid', 'gradient_plus_0.15'):
        data, signs, pairs = [], [], []
        for (case, name), chosen in average.items():
            if name != candidate:
                continue
            source = average[(case, 'source')]
            delta = chosen['G']-source['G']
            dq = np.asarray(chosen['Q'])-source['Q']
            data.append({'case_id': case, 'map_seed': chosen['map_seed'], 'delta_G': delta, 'delta_Q': dq.tolist()})
            signs.append(delta)
            if abs(delta) > .005:
                pairs.append({'map_seed': chosen['map_seed'], 'correct': ((dq*delta > 0) & (np.abs(dq)>1e-6)).tolist()})
        outputs[candidate] = {'states': len(data), 'positive': int(sum(d>.005 for d in signs)),
            'negative': int(sum(d<-.005 for d in signs)), 'near_tie': int(sum(abs(d)<=.005 for d in signs)),
            'mean_delta_G': float(np.mean(signs)), 'q1_sign_agreement': float(np.mean([p['correct'][0] for p in pairs])) if pairs else None,
            'q1_map_bootstrap_95ci': map_bootstrap(pairs, 0), 'paired_rows': data,
            'role': 'Source-vs-selected improvement control; differs from plus-vs-minus main gradient acceptance.'}
    return outputs


def review(source, destination):
    manifest = json.loads((source/'audit/manifest.json').read_text())
    if manifest['production_code_sha256'] != production_hashes():
        raise RuntimeError('Production inputs changed from original critic audit')
    cfg = json.loads((source/'configuration.json').read_text())
    for step in AUDIT_STEPS:
        p = Path(cfg['checkpoint_directory'])/f'model_{step:06d}_full_snapshot.pt'
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest['checkpoint_sha256'][str(step)]:
            raise RuntimeError(f'Original critic snapshot changed: {p}')
    collected, provenance = {}, []
    for step, phase in itertools.product(AUDIT_STEPS, ('deterministic', 'noise')):
        name = f'step_{step:06d}_{phase}.jsonl'
        collected[name], info = unique_rows(source/'audit'/name, 1248 if phase=='deterministic' else 936)
        provenance.append(info)
    # Check completeness before creating the result, so incomplete data never
    # masquerade as a complete review or consume the intended destination.
    (destination/'audit').mkdir(parents=True, exist_ok=False)
    dump(destination/'audit/manifest.json', manifest)
    for name, rows in collected.items():
        (destination/'audit'/name).write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    dump(destination/'provenance.json', {'sources': provenance, 'originals_rewritten': False,
        'dedup_key': ['step', 'case_id', 'candidate', 'replica'], 'conflicting_duplicates': 0,
        'note': 'Raw snapshots may later gain identical appended rows; the review uses these captured complete snapshots.'})
    result = summarize(destination)
    noise = {str(step): noise_changes(collected[f'step_{step:06d}_noise.jsonl']) for step in AUDIT_STEPS}
    dump(destination/'noise_changes.json', noise)
    assert all(result['steps'][str(step)]['target_noise_control']['complete'] for step in AUDIT_STEPS)
    lines = ['# Full Critic 20k完整验收', '',
        '**主验收未通过；目标平滑噪声对照已经补齐。** 这是旧7m、12s任务的冻结源Actor实验，与新16m/40s任务分开解释。', '',
        '累计20k含4k前缀与16k新增Full转移；Actor及其Target更新0，Critic累计19489次。固定seeds80–89、Stage0/3共78个状态。每个checkpoint：1248条确定性记录，936条噪声记录（78状态×3候选×4共同噪声副本）。', '',
        '| 步数 | 确定性源Q1 MAE | 噪声源Q1 MAE | Q1主排序 | Q1主梯度一致率 | 第一层 | 第二层 |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for step in AUDIT_STEPS:
        r = result['steps'][str(step)]
        lines.append(f'| {step} | {r["source_calibration"]["q1"]["mae"]:.3f} | {r["target_noise_control"]["source_calibration"]["q1"]["mae"]:.3f} | {r["ranking"]["q1"]["agreement"]:.1%} | {r["gradient"]["0.15"]["agreement"]:.1%} | {r["verdict"]["level1_value_pass"]} | {r["verdict"]["level2_status"]} |')
    lines += ['', '## 四副本共同噪声对照', '',
        '第一步候选不加噪声，后续固定源Actor加原TD3目标平滑sigma0.2/clip0.5；相同状态/副本使用相同噪声。此对照检查确定性G与平滑策略价值定义的差异，不替代主验收，也不是高精度MC估计。以下比较候选与源动作，近似平局阈值为缩放G差0.005。', '',
        '| 步数 | 候选 | 回报改善 / 下降 / 近似平局 | 平均ΔG | Q1预测符号一致率 |',
        '| --- | --- | --- | ---: | ---: |']
    for step in AUDIT_STEPS:
        for candidate, label in (('q1_best_grid', 'Q1最高网格'), ('gradient_plus_0.15', '正梯度0.15')):
            n = noise[str(step)][candidate]
            rate = 'NA' if n['q1_sign_agreement'] is None else f'{n["q1_sign_agreement"]:.1%}'
            lines.append(f'| {step} | {label} | {n["positive"]} / {n["negative"]} / {n["near_tie"]} | {n["mean_delta_G"]:+.3f} | {rate} |')
    lines += ['', '## 证据与边界', '',
        '已证实：延长同机制训练改善部分Q校准，但20k仍未达到MAE≤2、abs(bias)≤1；主排序未达到80%，主要局部梯度未达到70%及地图聚类CI门槛。价值绝对误差与动作方向独立判断，不能仅凭TD误差下降或平均候选收益为正解冻Actor。', '',
        '较强判断：当前Critic尚不适合作为可靠的Actor改进依据。尚未证明具体根因是Replay覆盖、成功经验组成、目标平滑、折扣传播或函数逼近中的哪一个。主实验同时增加数据与优化次数，不能单独归因为经验不足。', '',
        '重复记录处理：恢复审计期间发现并发追加的完全相同记录。本工具按实验键去重，逐条验证重复内容完全一致；冲突数为0。原JSONL、checkpoint及训练产物未删除或覆盖，重复行不计作独立证据。完整来源快照指纹和数量见[provenance.json](provenance.json)。', '',
        '下一步：先在新16m任务的独立分支验证局部Critic校准和Actor能力保留，不继续无条件延长原训练。对应新实验见[综合执行报告](../../full_ego20x10_next_step_v1/REPORT.zh.md)。', '',
        '机器可读：[完整两层验收](audit/summary.json)、[噪声配对变化](noise_changes.json)。原协议：[工作日志](../../../docs/work_logs/2026-10-13_full_critic_frozen_v1_protocol.md)。']
    (destination/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    print(f'[Full acceptance] {destination}/REPORT.zh.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT/'completed_review')
    args = parser.parse_args()
    review(args.source, args.output)


if __name__ == '__main__':
    main()
