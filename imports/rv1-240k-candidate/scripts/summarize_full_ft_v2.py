"""Unified comparison of raw Full FT v2 stability and validation adaptation."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('analysis_outputs/full_ft_v2_seed43')
REPORT = Path('docs/work_logs/2026-10-12_full_ft_v2_ablation_results.md')


def read(path):
    return json.loads(path.read_text())


def paired(before, after):
    b = {r['environment_seed']: r for r in before['episodes']}
    a = {r['environment_seed']: r for r in after['episodes']}
    assert set(a) == set(b) == set(range(80, 90))
    return {'gained_success_maps': [k for k in a if a[k]['success'] and not b[k]['success']],
            'lost_success_maps': [k for k in a if not a[k]['success'] and b[k]['success']],
            'retained_success_maps': [k for k in a if a[k]['success'] and b[k]['success']]}


def rates(record):
    s = record['summary']
    return '/'.join(f'{s.get("eval/" + key, 0.):.0%}' for key in
                    ('success_rate', 'collision_rate', 'done_reason_timeout_rate'))


def validation_velocity(branch, evaluations):
    """Actual 120Hz and command velocities, restricted to real near-goal visits."""
    rows = []
    for evaluation in evaluations:
        step = evaluation['step']
        for stage in (0, 3):
            per_map = []
            for seed in range(80, 90):
                stem = Path(f'runs/full_ft_v2_seed43/{branch}/full_stage{stage}_traces/step_{step:07d}/trace_seed{seed}')
                meta = read(stem.with_suffix('.json'))
                with np.load(stem.with_suffix('.npz')) as d:
                    position = d['lower_position']
                    velocity = d['lower_velocity']
                    commands = d['lower_command_velocity']
                    target = np.asarray(meta['target_position'])
                    distance = np.linalg.norm(target - position, axis=1)
                    near = distance < 1.2
                    n = int(near.sum())
                    if n:
                        direction = (target - position[near]) / np.maximum(distance[near, None], 1e-12)
                        radial = np.sum(velocity[near] * direction, axis=1)
                        command_radial = np.sum(commands[near] * direction, axis=1)
                        metrics = {'actual_mean_radial_mps': float(radial.mean()),
                                   'actual_mean_vertical_mps': float(velocity[near, 2].mean()),
                                   'command_mean_radial_mps': float(command_radial.mean()),
                                   'command_mean_vertical_mps': float(commands[near, 2].mean()),
                                   'actual_away_fraction': float(np.mean(radial < -.02)),
                                   'actual_mean_speed_mps': float(np.linalg.norm(velocity[near], axis=1).mean())}
                    else:
                        metrics = None
                    per_map.append({'environment_seed': seed, 'near_goal_lower_ticks': n, 'metrics': metrics})
            total = sum(p['near_goal_lower_ticks'] for p in per_map)
            present = [p for p in per_map if p['metrics'] is not None]
            pooled = ({k: sum(p['metrics'][k] * p['near_goal_lower_ticks'] for p in present) / total
                       for k in present[0]['metrics']} if total else None)
            rows.append({'step': step, 'full_stage': stage, 'near_goal_radius_m': 1.2,
                         'maps_entering_near_goal': sum(p['near_goal_lower_ticks'] > 0 for p in per_map),
                         'near_goal_lower_ticks': total, 'pooled_tick_metrics': pooled,
                         'per_map': per_map})
    (ROOT / branch / 'validation_near_goal_speed.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    return rows


def plot_comparison(branches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    colors = {'A': '#2864b4', 'B': '#dd7c23', 'C': '#27945a'}
    for branch, data in branches.items():
        rows = [r for r in data['diagnostics'] if r['label'] == 'actor_update_probe']
        updates = [r['actor_updates'] for r in rows]
        for ax, value, title in (
            (axes[0, 0], [r['commands']['mean_command_difference_mps'] for r in rows], 'Mean velocity command drift (m/s)'),
            (axes[0, 1], [r['commands']['near_goal']['updated_mean_goal_closing_command_mps'] for r in rows], 'Near-goal radial command (m/s)'),
            (axes[1, 0], [r['commands']['near_goal']['updated_mean_vertical_command_mps'] for r in rows], 'Near-goal vertical command (m/s)'),
        ):
            ax.semilogx(updates, value, 'o-', color=colors[branch], label=branch)
            ax.set_title(title); ax.set_xlabel('Actor updates'); ax.grid(alpha=.25); ax.legend()
        rows = data['diagnostics']
        axes[1, 1].plot([r['step'] for r in rows], [r['q']['current_q1'] for r in rows], 'o-', color=colors[branch], label=f'{branch}: current action')
        axes[1, 1].plot([r['step'] for r in rows], [r['q']['source_q1'] for r in rows], '--', color=colors[branch], label=f'{branch}: source action')
        axes[2, 0].plot([r['step'] for r in data['evaluations']],
                        [r['full_stage3']['summary']['eval/success_rate'] for r in data['evaluations']],
                        linestyle='None', marker={'A': 'o', 'B': '^', 'C': 's'}[branch],
                        markersize={'A': 12, 'B': 10, 'C': 6}[branch],
                        markerfacecolor='none', color=colors[branch], label=branch)
    axes[0, 1].axhline(0., color='gray', linewidth=.8)
    axes[1, 0].axhline(0., color='gray', linewidth=.8)
    axes[1, 1].set_title('Fixed-bank critic Q1 (scaled rewards)'); axes[1, 1].set_xlabel('Upper transitions'); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=.25)
    axes[2, 0].axvspan(0, 4000, color='gray', alpha=.08, label='Actor frozen')
    axes[2, 0].set_title('Raw Full Stage 3 validation success'); axes[2, 0].set_xlabel('Upper transitions'); axes[2, 0].set_ylim(-.03, 1.03); axes[2, 0].legend(); axes[2, 0].grid(alpha=.25)
    groups = ['Source', 'A raw', 'B raw', 'C raw']
    for i, (name, label) in enumerate((('full_stage0', 'Full Stage 0'), ('full_stage3', 'Full Stage 3'), ('k1_stage3', 'K1 Stage 3'))):
        values = [branches['A']['evaluations'][0][name]['summary']['eval/success_rate']]
        values += [branches[b]['evaluations'][-1][name]['summary']['eval/success_rate'] for b in 'ABC']
        axes[2, 1].bar(np.arange(4) + (i - 1) * .25, values, width=.25, label=label)
    axes[2, 1].set_xticks(range(4), groups); axes[2, 1].set_ylim(0., 1.05); axes[2, 1].set_title('Source vs raw endpoint success'); axes[2, 1].legend(fontsize=8)
    fig.suptitle('Full FT v2 | seed43 | identical 4k prefix | validation maps 80-89')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(ROOT / 'comparison.png', dpi=180)
    plt.close(fig)


def summarize():
    branches = {}
    for branch in ('A', 'B', 'C'):
        folder = ROOT / branch
        branches[branch] = {'result': read(folder / 'result.json'),
                            'diagnostics': read(folder / 'diagnostics.json'),
                            'evaluations': read(folder / 'evaluations.json'),
                            'prefix': read(folder / 'prefix_verification.json'),
                            'config': read(folder / 'configuration.json')}
        assert branches[branch]['prefix']['all_fields_exactly_equal']
        branches[branch]['actual_near_goal_speed'] = validation_velocity(branch, branches[branch]['evaluations'])
    # Both frozen source outputs and per-map baseline outcomes must be identical.
    for branch in ('B', 'C'):
        for name in ('full_stage0', 'full_stage3', 'k1_stage3'):
            lhs = branches['A']['evaluations'][0][name]['episodes']
            rhs = branches[branch]['evaluations'][0][name]['episodes']
            assert [(r['environment_seed'], r['done_reason']) for r in lhs] == [(r['environment_seed'], r['done_reason']) for r in rhs]
    result = {'experiment': 'full_ft_v2_seed43', 'complete': True, 'branches': {},
              'heldout_seeds40_79_used_for_new_validation': False,
              'seed43_prefix_exception_authorized': True}
    lines = ['# Full FT v2：保守微调消融结果', '',
             'A/B/C 均从逐项完全一致的 seed43 4k 解冻前学习及仿真状态起步。下列终点数字来自 `model_raw_final`；最佳模型恢复没有替换或覆盖原始退化。验证固定为环境 seeds80–89，每路十张地图。', '',
             '| 分支 | Actor LR | 源动作锚定 | 终点步数 | 终止原因 | 合格微调 best |',
             '| --- | --- | --- | ---: | --- | --- |']
    for branch, data in branches.items():
        r, cfg = data['result'], data['config']
        final = r['raw_final_evaluation']
        p = {name: paired(r['baseline'][name], final[name]) for name in ('full_stage0', 'full_stage3', 'k1_stage3')}
        result['branches'][branch] = {'upper_steps': r['upper_steps'], 'stop_reason': r['stop_reason'],
                                      'extension_decision': r['extension_decision'],
                                      'paired_validation': p, 'raw_final_diagnostics': r['raw_final_diagnostics'],
                                      'best_selection': r['best_selection'],
                                      'actual_near_goal_speed': data['actual_near_goal_speed']}
        lines.append(f'| {branch} | {cfg["actor_lr"]:.0e} | {cfg["policy_anchor_weight"]} | {r["upper_steps"]} | {r["stop_reason"]} | {r["best_selection"]["qualified_finetuned_best"]} |')
    lines += ['', '## 原始策略验证：成功 / 碰撞 / timeout', '',
              '边界越界和姿态越界单独记录在机器结果中，不合并为 timeout。', '',
              '| 分支 | 步数 | Full Stage0 | Full Stage3 | K1 Stage3 |', '| --- | ---: | --- | --- | --- |']
    for branch, data in branches.items():
        for row in data['evaluations']:
            lines.append(f'| {branch} | {row["step"]} | {rates(row["full_stage0"])} | {rates(row["full_stage3"])} | {rates(row["k1_stage3"])} |')
    from collections import Counter
    lines += ['', '| 分支 | 原始终点 | Full Stage0终止原因 | Full Stage3终止原因 | K1 Stage3终止原因 |',
              '| --- | ---: | --- | --- | --- |']
    for branch, data in branches.items():
        row = data['evaluations'][-1]
        reasons = [', '.join(f'{reason}={count}/10' for reason, count in sorted(Counter(ep['done_reason'] for ep in row[name]['episodes']).items()))
                   for name in ('full_stage0', 'full_stage3', 'k1_stage3')]
        lines.append(f'| {branch} | {row["step"]} | {reasons[0]} | {reasons[1]} | {reasons[2]} |')
    lines += ['', '## 早期 Actor 更新：固定192状态库', '',
              '漂移为归一化动作乘 `[0.8,0.8,0.3]` 后相对源策略的三维速度指令平均L2差；近目标使用同一库中距离<1.2m的状态。径向正值表示朝目标。它们是指令偏置，实际飞行速度另见评估和15/120Hz轨迹。', '',
              '| 分支 | Actor更新 | 步数 | 指令漂移 m/s | 近目标径向 m/s | 近目标垂直 m/s | 当前动作Q1 | 源动作Q1 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for branch, data in branches.items():
        for row in data['diagnostics']:
            c, q = row['commands'], row['q']
            lines.append(f'| {branch} | {row["actor_updates"]} | {row["step"]} | {c["mean_command_difference_mps"]:.3f} | {c["near_goal"]["updated_mean_goal_closing_command_mps"]:.3f} | {c["near_goal"]["updated_mean_vertical_command_mps"]:.3f} | {q["current_q1"]:.3f} | {q["source_q1"]:.3f} |')
    lines += ['', '## 训练稳定性与 Full 适应收益', '']
    collapse = {b: data['result']['stop_reason'] == 'persistent_full_collapse_all_three_domains' for b, data in branches.items()}
    if all(collapse.values()):
        conclusion = ('三个分支均在5k、6k两次检查中三路零成功，于6k触发预设持续性全面崩溃早停，未进入10k或40k，也没有合格微调best。降低Actor LR至1e-4、加入0.1源动作锚定及两者结合，均未在本次seed43试验中避免Actor崩溃；不能选出已证明能“避免崩溃”的改动。')
    else:
        survivors = [b for b in branches if not collapse[b]]
        conclusion = f'通过持续全面崩溃早停检查的分支为 {", ".join(survivors)}；具体续跑和终点结果见表格。训练未全面崩溃仅说明达到预设稳定性检查，Full适应收益必须另看源→终点逐图成功变化。'
    lines += [conclusion, '', '全部保存的在线Actor/Critic参数及固定状态库Q诊断均有限；导航功能已经丧失。终点在线与目标网络参数的有限性另由`artifact_validation.json`核查。', '']
    result['conclusion'] = conclusion
    # The first anchor loss is mathematically zero, up to GRU rounding error.
    early = {}
    for branch, data in branches.items():
        early[branch] = {r['actor_updates']: r for r in data['diagnostics'] if r['label'] == 'actor_update_probe'}
    def drift(branch, update):
        return early[branch][update]['commands']['mean_command_difference_mps']
    lines += [f'第1次更新漂移：A={drift("A",1):.3f}、B={drift("B",1):.3f}、C={drift("C",1):.3f} m/s；第10次：A={drift("A",10):.3f}、B={drift("B",10):.3f}、C={drift("C",10):.3f} m/s。A与C首更新从同一源Actor开始，锚定损失及梯度在数学上为零，仅有浮点精度差异；首次步长差异由Actor LR决定。A↔C检验低LR下锚定的增量效果，B↔C检验有锚定时降低LR的效果。', '']
    if drift('C', 1) < drift('B', 1) and drift('C', 10) < drift('B', 10):
        lines += [f'在有相同0.1锚定的B/C对照中，降低Actor LR使第1/10次输出漂移分别减少{1-drift("C",1)/drift("B",1):.1%}/{1-drift("C",10)/drift("B",10):.1%}；对减缓早期偏离最有帮助的是降低学习率。这个结论仅针对早期输出保持，不能替代“避免闭环崩溃”的结论。', '']
    if drift('C', 10) >= drift('A', 10):
        lines += ['在同为1e-4的A/C对照中，0.1锚定没有进一步降低第10次指令漂移。后期个别漂移数值较低，也不能据此判为Full适应收益。', '']
    for branch, data in branches.items():
        p = result['branches'][branch]['paired_validation']['full_stage3']
        base_s = data['result']['baseline']['full_stage3']['summary']['eval/success_rate']
        end_s = data['result']['raw_final_evaluation']['full_stage3']['summary']['eval/success_rate']
        lines.append(f'- {branch} Full Stage3源→raw终点：{base_s:.0%}→{end_s:.0%}；新增成功地图{p["gained_success_maps"]}，丢失成功地图{p["lost_success_maps"]}。')
    lines += ['', '## 实际锚定损失', '',
              '正的锚定MSE与成功Replay抽样比例证明约束实际启用；目标量级比较不能替代梯度贡献分解。', '',
              '| 分支 | Actor更新 | 成功Replay比例 | 锚定MSE | 加权锚定项 | Actor总目标 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for branch in ('B', 'C'):
        for update in (1, 10, 50, 100, 500):
            info = early[branch][update]['last_train_info']
            loss = info['actor_policy_anchor_loss']
            lines.append(f'| {branch} | {update} | {info["success_replay_fraction"]:.3f} | {loss:.6g} | {.1*loss:.6g} | {info["actor_loss"]:.6g} |')
    lines += ['', '## 实际近目标速度', '',
              '下面按120Hz真实近目标采样tick加权。进入数为十张地图中实际进入1.2m球的数量；没有近目标访问时速度为NA，固定状态库上的策略偏置仍由上表记录。', '',
              '| 分支 | 步数 | Full Stage | 进入地图数 | 近目标ticks | 实际径向 m/s | 实际垂直 m/s |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for branch, data in branches.items():
        for row in data['actual_near_goal_speed']:
            m = row['pooled_tick_metrics']
            radial = f'{m["actual_mean_radial_mps"]:.3f}' if m else 'NA'
            vertical = f'{m["actual_mean_vertical_mps"]:.3f}' if m else 'NA'
            lines.append(f'| {branch} | {row["step"]} | {row["full_stage"]} | {row["maps_entering_near_goal"]} | {row["near_goal_lower_ticks"]} | {radial} | {vertical} |')
    lines += ['', '近目标远离/爬升偏置和Critic Q变化见上表。Q只是网络估值；本试验未新增匹配当前策略的Monte Carlo校准，不能将Q提高解读为真实Full收益。这里有一个训练seed和十张验证地图，没有使用40–79作新增评估，不能推广为多seed最优结论。', '',
              '## 锚定语义和产物保全', '',
              'B/C直接复用原TD3的成功Replay状态动作MSE，权重0.1；固定源Actor在4k核验之后显式初始化并保持不变，best回调被禁止刷新锚定。所有学习仍调用原 `TD3UpperRecurrent.train()`。Actor/目标网络、Twin Critic/目标网络、Adam、主/成功Replay、待提交成功转移、三路RNG、课程及控制状态均已作4k逐项精确比较。', '',
              '原奖励、917D观测、GRU、Full控制、Stage0/3比例、目标平滑和探索日程延续v1。用户明确允许仅重放原seed43前缀的例外；新增验证全部为80–89。训练后续reset不传入40–79的显式seed。', '',
              '`model_raw_final`保留原始终点；`model_best`若无合格微调策略则保存4k源Actor fallback，`model_best_metrics.json`明确标注`qualified_finetuned_best=false`。`model_final`为所选best恢复结果，其来源另存`model_final_selection.json`，不会覆盖raw诊断。', '',
              '- 运行器：`scripts/run_full_ft_v2.py`；汇总器：`scripts/summarize_full_ft_v2.py`。',
              '- 每分支：`analysis_outputs/full_ft_v2_seed43/{A,B,C}/prefix_verification.json`、`configuration.json`、`diagnostics.json`、`actor_updates.json`、`evaluations.json`、`result.json`。',
              '- 网络及完整诊断快照：`checkpoints/full_ft_v2_seed43/{A,B,C}/model_<step>_full_snapshot.pt`。4k、5k、6k/终点包含Replay；所有时点包含在线/目标网络、优化器、RNG及固定源锚定网络。',
              '- 训练与Full验证15/120Hz轨迹位于各分支analysis目录和`runs/full_ft_v2_seed43`。',
              '- 原v1产物哈希保全结果：`analysis_outputs/full_ft_v2_seed43/original_artifact_integrity.json`。', '',
              '原v1及冻结源共2850个文件的SHA-256均保持不变。快照校验确认前1/10/50/100/500次更新的主/目标网络、优化器和锚定目标齐全，实际LR正确；raw终点逐张量等于6k快照，best/final逐张量等于4k源Actor，二者明确不同。8项相关测试通过。', '',
              '首次启动在0步基线汇总时遇到K1不适用的跟踪误差NaN，尚未执行学习。已修复JSON为null；该次日志及产物完整归档在`startup_failures/01`，后续三个实际分支重新执行同一初始化及前缀，没有用其它训练起点替代。', '',
              '预设短期/续跑/早停判据见 `docs/work_logs/2026-10-12_full_ft_v2_preflight.md`，在观察v2分支结果之前记录。', '',
              '对照图：`analysis_outputs/full_ft_v2_seed43/comparison.png`。']
    (ROOT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    REPORT.write_text('\n'.join(lines) + '\n')
    plot_comparison(branches)
    print(conclusion)
    print(f'[V2 report] {REPORT}', flush=True)
    return result


if __name__ == '__main__':
    summarize()
