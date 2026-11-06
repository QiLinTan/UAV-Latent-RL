"""Report completed acceptance, task probes and short isolated mechanisms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.audit_ego20x10_value import write
from scripts.run_full_ft_v2 import sha

ROOT = Path('analysis_outputs/full_ego20x10_next_step_v1')
TASK = Path('analysis_outputs/full_ego20x10_contract_corrections_v1')
GUARD = Path('analysis_outputs/full_ego20x10_actor_guard_pilot_v1')
LOCAL = Path('analysis_outputs/full_ego20x10_local_calibration_pilot_v1')
OLD = Path('analysis_outputs/full_critic_frozen_v1_seed43/completed_review')


def read(path):
    return json.loads(path.read_text())


def outcomes(episodes):
    return {'episodes': len(episodes), 'success': sum(e['success'] for e in episodes),
        'outcomes': dict(Counter(e['done_reason'] for e in episodes)),
        'mean_minimum_distance': float(np.mean([e['minimum_goal_distance'] for e in episodes])),
        'mean_tracking_error': float(np.mean([e['tracking_error_mean'] for e in episodes])),
        'min_tree_clearance': float(min(e['minimum_tree_clearance'] for e in episodes))}


def plots(root, validation, probes, encoding, discounts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    counts = [0, 1, 10, 50, 100]
    for label, color in (('baseline', '#bd3b39'), ('step_guard', '#167aad')):
        success = [sum(e['success'] for e in validation['initial']['L0'])]
        success += [sum(e['success'] for e in validation[f'{label}_actor{n}']['L0']) for n in counts[1:]]
        axes[0].plot(range(5), np.array(success)/20, marker='o', color=color, label=label)
        chosen = {r['actor_proposals']: r for r in probes[label]}
        axes[1].plot(range(5), [chosen[n]['cumulative_change_mean_mps'] for n in counts], marker='o', color=color, label=label)
    axes[0].set(xticks=range(5), xticklabels=counts, xlabel='Actor proposals', ylabel='Paired L0 success rate', ylim=(-.04, 1.04))
    axes[1].set(xticks=range(5), xticklabels=counts, xlabel='Actor proposals', ylabel='Mean drift from initial action (m/s)')
    times = np.linspace(0, 40, 401)
    for gamma in (.99, .997):
        axes[2].plot(times, gamma**(times*15), label=f'gamma={gamma}')
    axes[2].set(xlabel='Future time (seconds)', ylabel='Discount weight', ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.22)
        axis.legend()
    fig.suptitle('Separate engineering candidates from demonstrated learning improvements')
    fig.savefig(root/'mechanism_and_horizon.png', dpi=180)
    plt.close(fig)


def report(root):
    for folder in (TASK, GUARD, LOCAL):
        assert read(folder/'protection_result.json')['unchanged']
    old = read(OLD/'audit/summary.json')
    assert all(s['target_noise_control']['complete'] for s in old['steps'].values())
    root.mkdir(parents=True, exist_ok=False)
    encoding = read(TASK/'encoding.json')
    discounts = read(TASK/'discount.json')
    physical = read(TASK/'physical_proof.json')
    rates = read(TASK/'guidance_replay_rates.json')
    validation = read(GUARD/'validation.json')
    probes = {label: read(GUARD/label/'checkpoint_probes.json') for label in ('baseline', 'step_guard')}
    local = read(LOCAL/'pair_scores.json')
    gradients = read(LOCAL/'holdout_gradient_branches.json')
    gradient_stats = {}
    for label in ('td_only', 'td_local'):
        selected = [r for r in gradients if r['branch']==label]
        valid = [r for r in selected if r['distinguishable']]
        gradient_stats[label] = {'states': len(selected), 'distinguishable_states': len(valid),
            'positive': sum(r['direction_agrees'] for r in valid),
            'mean_G_plus_minus_G_minus': float(np.mean([r['G_plus_minus_G_minus'] for r in selected]))}
    summary = {'old7m_acceptance': old,
        'actor_guard': {'training': read(GUARD/'training_proof.json'),
            'paired_evaluation': {k: {task: outcomes(es) for task, es in groups.items()} for k, groups in validation.items()},
            'probes': probes},
        'contract': {'encoding': encoding, 'discount': discounts, 'guidance_rates': rates, 'physical': physical},
        'local_calibration': {'scores': local, 'gradient_checks': gradient_stats,
            'training': read(LOCAL/'training_proof.json')},
        'decision': {'resume_large_training': False, 'adopt_step_guard_alone': False,
            'adopt_local_calibration': False, 'formal_contract_changed': False,
            'new_task_learning_comparison_completed': False,
            'reason': 'Local direction evidence insufficient and basic navigation preservation failed. Task candidates passed engineering probes only.'},
        'verification': {'tests_passed': 17, 'new_online_learning_transitions': 0,
            'new_critic_updates': 800, 'new_actor_proposals': 200, 'sealed_test_episodes': 0},
    }
    protected = {}
    for folder in (TASK, GUARD, LOCAL):
        for path, value in read(folder/'protected_inputs.json').items():
            if path in protected:
                assert protected[path] == value
            protected[path] = value
    changed = [p for p, digest in protected.items() if sha(Path(p)) != digest]
    assert not changed, changed
    summary['verification']['protected_files_unchanged'] = len(protected)
    write(root/'summary.json', summary)
    write(root/'protection_result.json', {'unchanged': True, 'checked_files': len(protected), 'changed': []})
    plots(root, validation, probes, encoding, discounts)
    lines = ['# Full EGO20x10：验收与下一步机制执行结果', '',
        '**本轮完成了20k Critic完整验收、16m任务单因素工程验证、Actor单步保护短对照和冻结Actor的局部Critic校准短对照。当前两个训练机制候选都未满足进入大规模训练的条件。** 正式模式、奖励、GRU、课程、checkpoint保持原样，封存test40没有执行。', '',
        '## 执行范围', '',
        '旧7m/12s实验使用已有4k/10k/20k快照，没有重新训练20k；补齐与合并目标噪声控制。新16m/40s机制实验从原5k完整学习状态出发，使用原A真实Replay前缀与录制采样索引/目标噪声，各分支计算自己的TD target，不产生新在线训练经验。Actor保护两分支各200次Critic更新、100次Actor候选更新；校准两分支各200次Critic更新、Actor0次。', '',
        '“不等待Critic完全正确”在这里体现为独立测试Actor步长保护，而不是在Critic方向未通过时继续长训。任务候选只做编码、奖励和物理一致性验证，本轮没有冒充完成其学习收益对照，也没有把不同输入编码喂给旧Actor比较成功率。', '',
        '## 1. 旧7m Critic：绝对价值改善，局部指导未通过', '',
        '| 累计步数 | 源Q1 MAE（确定性） | 源Q1 MAE（共同噪声） | 主动作排序一致率 | 主梯度一致率 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for step, r in old['steps'].items():
        lines.append(f'| {step} | {r["source_calibration"]["q1"]["mae"]:.3f} | {r["target_noise_control"]["source_calibration"]["q1"]["mae"]:.3f} | {r["ranking"]["q1"]["agreement"]:.1%} | {r["gradient"]["0.15"]["agreement"]:.1%} |')
    lines += ['', '20k第一层和第二层均未通过。噪声四副本仅排查价值定义混淆，不能替代完整主梯度/排序验收。此结果属于旧7m任务，不与新16m初始Critic数值混用。[完整验收与去重来源](../full_critic_frozen_v1_seed43/completed_review/REPORT.zh.md)。', '',
        '## 2. Actor：限制每一步不等于保留导航能力', '',
        '采用固定188条原5k训练状态，以世界速度变化约束mean≤0.02m/s、P95≤0.05m/s；必要时对同一Adam梯度减小有效LR。拒绝时恢复Actor、Actor Target、Adam，保留Critic更新。验证地图不参与此步长判据。无保护分支最终四个网络逐张量等于原A5200，两个分支的采样索引、目标噪声及末尾Torch RNG一致。', '',
        '| Actor候选次数 | 原更新 L0成功/20 | 单步保护 L0成功/20 | 保护分支累计动作变化均值m/s |',
        '| --- | ---: | ---: | ---: |']
    guard_probes = {r['actor_proposals']: r for r in probes['step_guard']}
    for count in (0, 1, 10, 50, 100):
        before = validation['initial']['L0'] if count==0 else validation[f'baseline_actor{count}']['L0']
        after = validation['initial']['L0'] if count==0 else validation[f'step_guard_actor{count}']['L0']
        lines.append(f'| {count} | {sum(e["success"] for e in before)} | {sum(e["success"] for e in after)} | {guard_probes[count]["cumulative_change_mean_mps"]:.4f} |')
    transactions = [r['guard'] for r in read(GUARD/'step_guard/updates.json') if r['guard'] is not None]
    maximum_mean = max(r['accepted_change']['mean_mps'] for r in transactions)
    maximum_p95 = max(r['accepted_change']['p95_mps'] for r in transactions)
    lines += ['', f'100个候选都经回溯后接受；实际最大单步mean={maximum_mean:.5f}m/s、P95={maximum_p95:.5f}m/s，均在阈值内。第一步原mean=0.11357m/s，保护后0.01491m/s。一次更新后的12/20相对初始11/20属于小样本波动，不能称为学习提升。', '',
        '到100次时保护分支平均az约0.955，尚未达到|az|≥0.999的“饱和”阈值，但L0已16/20高度越界、4/20碰撞。原更新100次L0同样16高度越界、4碰撞。两者原始16m均0/20成功，19/20 XY越界、1/20碰撞。**仅记录饱和率不足以发现持续错误偏置；单步上限也不能阻止累计漂移。**', '',
        '边界：这是真实A行为Replay上的离线机制对照，数据包含A的坏行为；没有运行保护策略自己的在线采样。因此已证实“此共享数据与机制下单步保护不足”，尚未证明所有在线步长保护策略无效。2k/5k源Actor随机且相同，它的11/20 L0是初始行为，不代表已学会16m导航。', '',
        '## 3. Critic局部校准：拟合可改善，跨地图方向证据不足', '',
        '12个真实Full Replay状态，各轴使用源5k动作±0.15，共36对/72条真实Full分支。后续所有轴恢复同一固定5k Actor，仅第一动作不同；每条分支用fresh reset和原执行前缀，状态重建误差0，不使用save/restore。G为6s有限时域、γ0.99、缩放0.1。', '',
        '预先按地图划分：5张拟合train地图、7状态、21动作对；3张内部检查train地图、5状态、15动作对。没有validation/test监督。额外损失为100·ΣMSE(Qj(s,a+)−Qj(s,a−),G+−G−)，与原TD损失合并后每次只进行一次Critic Adam更新。权重是单一机制探针，不作参数搜索。TD-only四网络逐张量匹配原D3 B5200。', '',
        '| 分支 | 拟合Q1方向（可区分4对） | 内部检查Q1方向（可区分5对） | 拟合ΔG RMSE | 内部检查ΔG RMSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for label in ('td_only', 'td_local'):
        r = local[label][-1]
        lines.append(f'| {label} | {r["fit"]["q1"]["agreement"]:.0%} | {r["internal_holdout"]["q1"]["agreement"]:.0%} | {r["fit"]["q1"]["delta_return_rmse"]:.5f} | {r["internal_holdout"]["q1"]["delta_return_rmse"]:.5f} |')
    lines += ['', '最后再对5个内部检查状态，沿各自最终Q1梯度±0.15新执行20条Full分支。TD-only仅1对超过|ΔG|>0.005且方向错误；TD+local仅2对可区分，1对正确、1对错误。多数一动作局部干预的6s收益很小。**证据量不足，不能把40%相对20%当作校准通过。** 较大的近目标错误方向仍存在；拟合RMSE下降而内部检查RMSE略上升，也不支持直接采用。', '',
        '局部方向监督已实现且能改变训练，但当前样本覆盖及有限时域定义尚不能保证有效Actor梯度。6s G不是无限时域平滑策略Q真值；本轮没有解冻校准分支Actor，没有作“校准后导航改善”的结论。', '',
        '## 4. 16m任务：四项候选的工程效果已分别验证', '',
        '| 单因素候选 | 具体变化 | 证据与结论 |',
        '| --- | --- | --- |',
        '| goal_range | 目标差除[20,10,2] | 当前固定目标任务6000条Replay的目标X饱和20.32%→0%；从未clip的绝对位置恢复，历史不变 |',
        '| position_scale | x/10，y/5，(z−1)/1 | 位置块最大值19.93→1.75（高度允许到2.75，所以z不要求±1）；位置可精确反解，其他字段不变 |',
        '| guidance_project | 仅将引导期望速度投影到XY≤0.8、Z±0.3 | 消除不可执行目标；对相同Full轨迹只改变相关跟踪成本，不改变成功球/终止/物理 |',
        '| gamma997 | 0.99→0.997，每15Hz转移一次 | 半衰期4.60→15.38s；20s权重0.049→0.406；实际学习稳定性尚未测试 |', '',
        f'引导区2131真实状态中{rates["old_infeasible_fraction"]:.2%}的原期望Z超出命令上限，平均成本率{rates["old_tracking_rate"]["mean"]:.3f}→{rates["new_tracking_rate"]["mean"]:.3f}奖励单位/s；不是近目标全部成本被修复。', '',
        '五个候选（含baseline）分别执行16m起始短段、近目标上方、近目标下方，共15条Full空场物理诊断；相同脚本命令下，所有候选的逐tick位置、速度和动作逐元素相同，上层奖励与实际1–8下层tick求和误差0。上方入球控制的跟踪成本16.653→5.375，下方8.011→5.300；相同轨迹均成功。这些是控制/奖励探针，不是Actor成功率。', '',
        '另完成5候选×4障碍Stage的真实train森林reset/step检查，10/20/30/40树、917D、Full 15Hz/120Hz一致。保持40s、0.2m成功球、1.2m引导、1.75m雷达、两套课程、网络/GRU与正式配置。尚未选择组合版本：提高gamma可能增加目标幅值/方差，归一化变化也要求一致转换Replay或重新从零学习。', '',
        '![Actor能力保留、累计漂移和折扣](mechanism_and_horizon.png)', '',
        '## 5. 决策与下一步', '',
        '**已证实：** 旧任务Critic延长训练仍未提供合格局部方向；新任务共享Replay短对照里单步保护只能减缓最初变化；当前局部校准在拟合集改善而检查集未通过；16m四项候选的单因素工程语义已验证。', '',
        '**较强假设：** Actor需要同时限制相对保留策略的累计偏离，并在基本能力退化时回退；Critic方向监督需要更丰富的局部Full对照与不同运动状态覆盖。小型试验支持继续验证这些机制，不能保证其有效。', '',
        '**仍未验证：** 归一化、目标编码或更长折扣是否提升学习成功率；成功经验采样组成及坏行为Replay对方向错误的独立影响；6s方向与完整剩余40s/平滑策略方向能否一致。当前证据不支持马上改奖励权重或GRU，也不支持盲目再加训练预算。', '',
        '下一轮建议先做固定且一致编码的16m单因素学习对照；方向审计增加可区分样本量，按地图保留独立检查，并对接近终止、远目标与高度误差分组。Actor候选增加相对固定保留策略的累计约束，检查1/10/50/100次更新的L0及7/16m能力；Critic方向与导航能力都通过后再运行独立在线短段。当前**不启动10k/20k/240k正式续训**，不将任何候选注册为默认正式模式。', '',
        '## 复现与产物', '',
        '工程候选：[模块](../../scripts/ego20x10_contract_variants.py)、[物理探针](../../scripts/probe_ego20x10_contract_variants.py)；机制：[Actor短对照](../../scripts/run_ego20x10_actor_guard_pilot.py)、[Critic短对照](../../scripts/run_ego20x10_local_calibration_pilot.py)。入口均拒绝覆盖非空结果目录，新复现必须传入新的`--output`。', '',
        '```bash',
        'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/uav-mpl \\',
        '/home/tequila/miniconda3/envs/drones/bin/python -m scripts.probe_ego20x10_contract_variants \\',
        '  --output analysis_outputs/ego20x10_contract_new',
        '# 其他独立入口：run_ego20x10_actor_guard_pilot all / run_ego20x10_local_calibration_pilot',
        '# 每次指定一个新的--output，不把它们作为正式续训命令。',
        '```', '',
        f'17项测试通过，覆盖完整Actor/Target/Adam回退、无约束与原TD3一致、接受步长边界、局部权重0与冻结TD3逐张量一致、目标编码恢复、引导投影/终止不变、重复证据去重和冲突拒绝。{len(protected)}份原始输入/源码/checkpoint哈希未变；新在线训练转移0，test40执行0。', '',
        '结果：[任务工程数据](../full_ego20x10_contract_corrections_v1/protocol.json)、[Actor配对验证](../full_ego20x10_actor_guard_pilot_v1/validation.json)、[Critic梯度分支](../full_ego20x10_local_calibration_pilot_v1/holdout_gradient_branches.json)、[综合机器数据](summary.json)。', '',
        '过程中的`*_precheck`目录保留，但不参与最终证据：一次引导日志float32一致性修复，一次CPU线程数1与原A线程2的精确复现失败。最终目录通过相应一致性检查；未修改生产算法来绕过检查。']
    (root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    for folder, source in ((TASK, '/tmp/ego20x10_contract_probe.log'),
                           (GUARD, '/tmp/ego20x10_guard_pilot.log'),
                           (LOCAL, '/tmp/ego20x10_local_calibration.log')):
        shutil.copyfile(source, folder/'console.log')
    print(f'[Report] {root}/REPORT.zh.md', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    report(args.output)


if __name__ == '__main__':
    main()
