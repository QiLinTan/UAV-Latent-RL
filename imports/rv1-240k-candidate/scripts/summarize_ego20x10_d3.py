"""Paired D3 metrics, scientific figures, decision record and Chinese report."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, rankdata

from scripts.run_ego20x10_d3 import ROOT, BASELINE, sha


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def episode_summary(episodes):
    return dict(episodes=len(episodes),outcomes={r:sum(e['done_reason']==r for e in episodes) for r in ('success','collision','height_bound','xy_bound','timeout')},
                arrivals={str(radius):sum(e['radius_arrivals'][str(radius)]['reached'] for e in episodes) for radius in (.2,.3,.5,1.2)},
                **{k:float(np.mean([e[k] for e in episodes])) for k in ('minimum_goal_distance','final_goal_distance','tracking_error_mean',
                   'actual_speed_mean','minimum_tree_clearance','duration_s','vertical_action_mean','vertical_saturation_fraction')})


def paired_maps(initial,candidate):
    comparisons=[]
    for a,b in zip(initial,candidate):
        assert a['map_id']==b['map_id'] and a['initial_position']==b['initial_position']
        comparisons.append(dict(map_index=a['map_index'],map_id=a['map_id'],initial_outcome=a['done_reason'],candidate_outcome=b['done_reason'],
             initial_success=a['success'],candidate_success=b['success'],minimum_distance_change=b['minimum_goal_distance']-a['minimum_goal_distance'],
             tracking_error_change=b['tracking_error_mean']-a['tracking_error_mean']))
    gains=sum(not r['initial_success'] and r['candidate_success'] for r in comparisons)
    losses=sum(r['initial_success'] and not r['candidate_success'] for r in comparisons)
    return dict(gains=gains,losses=losses,unchanged_success=sum(r['initial_success'] and r['candidate_success'] for r in comparisons),
                unchanged_failure=sum(not r['initial_success'] and not r['candidate_success'] for r in comparisons),
                exact_paired_discordance_p=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.,maps=comparisons)


def figures(updates,baseline,fine,ranking,values,probes):
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for source,label,color in ((baseline,'A original','C0'),(updates,'B critic lead 200','C1')):
        steps=[r['step'] for r in source]
        counts=[r['actor_updates'] if label.startswith('B') else (r['step']-4999)//2 for r in source]
        mean=[r['policy']['action_mean'][2] if label.startswith('B') else r['fixed_z_mean_after_update'] for r in source]
        sat=[r['policy']['z_saturation_fraction'] if label.startswith('B') else r['fixed_z_saturation_after_update'] for r in source]
        axes[0,0].plot(steps,mean,label=label,color=color);axes[0,1].plot(counts,mean,label=label,color=color)
        axes[1,0].plot(steps,sat,label=label,color=color);axes[1,1].plot(counts,sat,label=label,color=color)
    axes[0,0].set(title='Mean vertical action vs shared-data time',ylabel='Normalized az')
    axes[0,1].set(title='Mean vertical action vs Actor updates',ylabel='Normalized az')
    axes[1,0].set(title='Vertical saturation vs shared-data time',ylabel='Sz')
    axes[1,1].set(title='Vertical saturation vs Actor updates',ylabel='Sz')
    before=[r for r in fine if 'fixed_before_actor_update' in r]
    axes[2,0].plot([r['step'] for r in before],[r['fixed_before_actor_update']['q1_action_gradient_mean'][2] for r in before],label='A fixed states, before Actor')
    sampled={r['step']:r['policy'] for r in updates if 'q1_action_gradient_mean' in r['policy']}
    sampled.update({r['step']:r for r in probes})
    sampled=sorted(sampled.items())
    axes[2,0].plot([step for step,r in sampled],[r['q1_action_gradient_mean'][2] for step,r in sampled],label='B fixed states, after update')
    axes[2,0].axhline(0,color='gray',lw=.7);axes[2,0].set(title='Q1 vertical action gradient (different probe timing)',ylabel='dQ1 / daz')
    axes[2,1].plot([r['actor_updates'] for r in updates if r['actor_updated']],
                  [r['policy']['adjacent_policy_rms'] for r in updates if r['actor_updated']],color='C1',label='B per Actor update')
    axes[2,1].set(title='B policy change on 188 fixed states',ylabel='Adjacent action RMS')
    for i,ax in enumerate(axes.ravel()):ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Shared A transition step' if i%2==0 else 'Actor update count')
    axes[0,1].set_xlim(0,110);axes[1,1].set_xlim(0,110)
    fig.savefig(ROOT/'actor_mechanism.png',dpi=170);fig.savefig(ROOT/'actor_mechanism.pdf');plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(14,9),layout='constrained')
    for ax,record in zip(axes.ravel(),values):
        for label,color in (('A5000','C0'),('B5200','C1')):
            ax.plot([-1,-.5,0,.5,1],rankdata([v['q1'] for v in record['scans'][label]]),marker='o',label=label+' Q1',color=color)
        returns=[v['physical']['common']['returns']['90']['discounted_scaled_return'] for v in record['scans']['B5200']]
        ax.plot([-1,-.5,0,.5,1],rankdata(returns),linestyle='--',color='black',label='Common 6s return')
        ax.set(title=f'Index {record["index"]}: {"/".join(record["groups"])}',xlabel='Normalized first az',ylabel='Within-state rank',ylim=(.7,5.3));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.savefig(ROOT/'vertical_value_ranks.png',dpi=170);fig.savefig(ROOT/'vertical_value_ranks.pdf');plt.close(fig)


def main():
    protected=read(ROOT/'protected_inputs.json');assert all(sha(k)==v for k,v in protected.items())
    protocol=read(ROOT/'protocol.json');assert sha(ROOT/'protocol.json')==read(ROOT/'protocol_fingerprint.json')['sha256']
    updates=read(ROOT/'B/updates.json');base=read(BASELINE/'update_audit.json');fine=read(BASELINE/'fine_timing/update_audit.json')
    probes=read(ROOT/'B/checkpoint_probes.json');by_step={v['step']:v for v in probes}
    validation=read(ROOT/'validation.json');ranking=read(ROOT/'ranking_summary.json');values=read(ROOT/'paired_values.json')
    assert len(updates)==1000 and len(values)==12 and read(ROOT/'physical_proof.json')['new_full_branches']==240
    metrics={label:{task:episode_summary(eps) for task,eps in tasks.items()} for label,tasks in validation.items()}
    paired={label:{task:paired_maps(validation['A5000'][task],validation[label][task]) for task in ('L0','original16m')} for label in ('A6000','B5325','B5600','B6000')}
    paired['B6000_vs_A6000']={task:paired_maps(validation['A6000'][task],validation['B6000'][task]) for task in ('L0','original16m')}
    first_sat=next(r for r in updates if r['policy']['z_positive_saturation']==1)
    first09=next(r for r in updates if r['policy']['action_mean'][2]>=.9)
    timing=dict(first_positive_full_saturation_step=first_sat['step'],first_positive_full_saturation_actor_updates=first_sat['actor_updates'],
                first_mean_z09_step=first09['step'],first_mean_z09_actor_updates=first09['actor_updates'],baseline_full_saturation_actor_updates=63,
                baseline_full_saturation_step=5125,actor63=by_step[5325],final=by_step[6000])
    condition=(metrics['B6000']['L0']['outcomes']['success']>=9 and paired['B6000_vs_A6000']['L0']['gains']-paired['B6000_vs_A6000']['L0']['losses']>=5
              and by_step[5325]['z_saturation_fraction']<.5 and by_step[6000]['z_saturation_fraction']<.5)
    assert not condition  # This dataset is the no-benefit arm; do not launch conditional experiments.
    decisions=dict(C_executed=False,online_executed=False,formal_10k_20k_executed=False,test_policy_executions=0,
                   reason='B failed preregistered preservation criterion: L0=0/20, vertical saturation=100% at Actor63 and6k. No improvement to disentangle.',
                   critic_lead_200_sufficient=False,broader_critic_learning_hypothesis_disproved=False)
    write(ROOT/'validation_summary.json',metrics);write(ROOT/'paired_validation_changes.json',paired);write(ROOT/'collapse_timing.json',timing)
    write(ROOT/'decisions.json',decisions)
    alignment=[]
    for count in (22,50,63,100):
        b=next(r for r in updates if r['actor_updates']==count and r['actor_updated'])
        a=next((r['fixed_before_actor_update'] for r in fine if r['step']==5001+2*count),None)
        if count==100:a=next(r for r in read(BASELINE/'timeline_probes.json') if r['step']==5200)
        alignment.append(dict(actor_updates=count,A_policy=a,B_policy=b['policy'],A_step=4999+2*count,B_step=b['step'],
                              A_policy_source='Next pre-Actor probe has identical policy; Actor100 from saved5200 probe',
                              A_gradient_timing='Next Critic update; not directly aligned to B post-update gradient'))
    write(ROOT/'actor_count_alignment.json',alignment)
    guard=dict(unchanged_files=len(protected),unchanged=True,production_training_changes=0,A_retraining_updates=0,B_critic_updates_added=1000,B_actor_updates_added=400,
               test_policy_executions=0,C_executed=False,online_executed=False)
    write(ROOT/'protection_result.json',guard)
    figures(updates,base,fine,ranking,values,probes)
    table=[]
    for label in ('A5000','B5200','A6000','B5325','B5600','B6000'):
        for task in ('L0','original16m'):
            m=metrics[label][task];o=m['outcomes'];arr=m['arrivals']
            table.append(f'| {label} | {task} | {o["success"]}/20 | {arr["0.3"]}/20 | {arr["0.5"]}/20 | {o["collision"]} | {o["xy_bound"]} | {o["height_bound"]} | {o["timeout"]} | {m["minimum_goal_distance"]:.3f} | {m["tracking_error_mean"]:.4f} |')
    eval_table='\n'.join(table)
    rank_table='\n'.join(f'| {label} | {suffix} | {horizon/15:g}s | '+ ' | '.join(f'{ranking[label]["all"][suffix][str(horizon)][critic]["concordance"]:.1%}' for critic in ('q1','q2','q_min'))+' |' for label in ('A5000','A5200','B5200','B5325','A6000','B6000') for suffix in ('common','own') for horizon in (30,90))
    group_table='\n'.join(f'| {label} | {group} | {ranking[label][group]["common"]["90"]["q1"]["states"]} | {ranking[label][group]["common"]["30"]["q1"]["concordance"]:.1%} | {ranking[label][group]["common"]["90"]["q1"]["concordance"]:.1%} |' for label in ('A5000','B5200','B5325','B6000') for group in ('near','far','above','below'))
    aligned_table='\n'.join(f'| {r["actor_updates"]} | {r["A_step"]} | {r["A_policy"]["action_mean"][2]:+.6f} | {r["A_policy"]["z_saturation_fraction"]:.1%} | {r["B_step"]} | {r["B_policy"]["action_mean"][2]:+.6f} | {r["B_policy"]["z_saturation_fraction"]:.1%} | {r["B_policy"]["radial_command_mps"]:+.3f} |' for r in alignment)
    checkpoint_table='\n'.join(f'| {step} | {r["actor_updates"]} | {r["critic_updates"]} | {tuple(round(v,3) for v in r["action_mean"])} | {r["radial_command_mps"]:+.3f} | {r["q1"]["mean"]:+.3f} / {r["q2"]["mean"]:+.3f} | {r["q1_action_gradient_mean"][2]:+.5f} |' for step,r in by_step.items() if step in (5000,5043,5125,5200,5325,5600,6000))
    q_before=ranking['A5000']['all']['common']['90']['q1'];q_lead=ranking['B5200']['all']['common']['90']['q1']
    report=f'''# Full Scratch EGO20x10 — D3 Critic Lead 200

实验：`full_scratch_ego20x10_d3_critic_lead_200`，2026-10-13，seed43。完成共享A行为数据的受控离线B组5k→6k；A组复用D1精确训练复现。正式训练配置、奖励、GRU、网络结构、地图池、成功半径均未改，test40未用于策略执行。

**结论：本次200次Critic先行没有改善策略崩溃，也没有建立更合理的垂直动作排序。** B在第57次Actor更新时即对188个固定状态全部正向垂直饱和，A为第63次；B的第63次与6k仍100%正饱和，L0均0/20。Critic在Actor完全不动时的共同后续控制6s排序一致率Q1由{q_before['concordance']:.1%}降至{q_lead['concordance']:.1%}。这否定“当前协议下先学200次即可解决”的方案，不能否定所有Critic校准方法，也不能证明某一任务设计是唯一根因。

## 七个问题的直接回答

| 问题 | 结果 |
| --- | --- |
| Critic先行200次后排序改善吗？ | 没有总体改善。共同π5k、6s、12状态：Q1 54.9%→29.4%，Q2 45.1%→29.4%，min-Q 33.3%→29.4%；分组有升有降。 |
| B第63次Actor更新仍饱和吗？ | 是，5325步，188/188正向饱和，az均值约+0.99985。 |
| B6k仍崩溃吗？ | 是，400次Actor更新后仍188/188正饱和，az均值约+0.99997。 |
| L0能力保留或提高吗？ | 未保留。初始55%，B5325/B5600/B6000均0%。冻结阶段B5200保持55%仅因Actor未变。 |
| 是排序改善还是Actor更新减少？ | 没有改善需要归因；相同Actor次数下B更早饱和，不能把累计训练步后移188步当收益。 |
| 需要C Delay Only吗？ | 当前不需要；B没有达到事前“能力保留＋饱和降低”条件，无收益可分离。 |
| 值得进入短程真实在线验证吗？ | 当前方案不进入。完成诊断后停止，本轮无C、无在线6k/8k、无正式10k/20k。 |

## 1. 严格控制与训练预算

共同起点为原5k完整快照：Critic累计1次、Actor0次；Actor/target相同，Actor Adam尚无状态，Critic Adam保留已有一次更新状态。逐项保留原协议、三维Actor/双Critic/GRU、targets、Adam超参数、成功Replay297条、global RNG初态和课程/地图采样器来源。

| 项目 | A | B |
| --- | --- | --- |
| 原5k快照 | 同一份 | 同一份 |
| 5001–5200新增Critic/Actor | 200 / 100 | 200 / 0 |
| 5201–6000新增Critic/Actor | 800 / 400 | 800 / 400 |
| 6k累计Critic/Actor | 1001 / 500 | 1001 / 400 |

采样用A实际5001–6000主Replay前缀，所有main index均小于当时prefix size，没有未来数据。成功Replay在5k→6k期间没有新成功，297条数组完全相同。每批205主＋51成功，状态/动作/下一状态/已缩放奖励/终止mask及target noise共享。B使用自身actor_target/critic_target计算TD target，没有复制A的目标值。5001第一批B Critic loss和target均值与A完全相同；5200和6000实捕获输入/noise逐元素相同，TD target平均绝对差分别.01144、.05489，说明之后确实由不同网络产生。1000次Critic loss重构误差0，最终Torch smoothing RNG与A6k完全一致。

用**未改动的生产TD3.train**，只在前200次令隔离agent.actor_frozen=True。Critic target按total_it偶数做tau=.005 Polyak（先行阶段100次）；先行期间Actor和target Actor保持5k相同参数，Actor optimizer状态不变。第201次新增Critic更新时恢复标准联合更新（5201，Actor第1次）。测试验证偶数精确Polyak、奇数target不变。LR均3e−4、batch256、gamma.99、policy_freq2、noiseσ.2/clip.5、grad_clip_norm1.0不变。

原配置的policy_anchor_weight名义值0.1保留，但5k anchor对象为None，全程未建立/激活行为克隆锚定；实际anchor loss为0。未加载K1或Full微调权重。[事前协议](protocol.json)、[配置指纹](protocol_fingerprint.json)、[共享数据证明](shared_data_proof.json)、[训练证明](B/training_proof.json)。

离线分支不执行课程或采样新的行为Replay；课程和RNG初态由原5k来源封存，raw learner快照引用A的不可变Replay前缀和采样流。这些是共享数据诊断快照，不能冒充物理在线续训快照。真正在线分支须按原5k完整快照、新Full episode的续训语义另行启动。

## 2. 更新时序、三轴动作与梯度

每50次Critic更新及5043/5125/5200/5325等关键点保存raw四网络、优化器和计数；1000次记录固定188状态动作/Q分布。前100次Actor更新逐次记录局部三轴Q1/Q2/min梯度、logit、实际batch输出损失梯度/头部bias梯度及策略RMS。完整网络快照仅在关键/50步节点，并非每次Actor更新都保存网络。

| B上层数据步 | Actor次数 | Critic累计 | 动作均值(ax,ay,az) | 径向m/s | Q1/Q2均值 | 固定集dQ1/daz |
| --- | --- | --- | --- | --- | --- | --- |
{checkpoint_table}

先行结束5200，Actor仍是初始策略，但固定集平均dQ1/daz已经从−.00501变为+.03558；Q1对x/y的平均梯度为−.04702/−.04129。首个Actor更新5201的实际batch dQ1/da均值为(−.04970,−.04362,+.03491)，输出损失梯度为(+.000194,+.000170,−.000136)，头部bias梯度为(+.04363,+.04193,−.03485)。梯度下降正是把XY推向负侧、Z推向正侧；并不是物理探索分布突然改变，因为本阶段共享行为数据不受B Actor影响。

输出损失梯度从实际backward hook记录；头部参数梯度是经过原grad_clip_norm=1.0处理后、优化器使用的梯度。二者处于不同梯度空间，不能简单将某一bias梯度视作整个策略变化量。

固定集az均值在第14次Actor更新（5227）超过+.9，第57次（5313）首次188/188正向饱和；A相应为第55次及63次。B从无先负后正的明显下探，直接利用先行后Critic的正Z梯度。5325时径向平均命令−.318m/s，XY也退化，不是单轴问题。

| Actor次数 | A数据步 | A平均az | A垂直饱和率 | B数据步 | B平均az | B垂直饱和率 | B径向m/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
{aligned_table}

按数据步对齐：A5125已饱和，B此时没有Actor更新；B首次全饱和5313比A5125晚188个数据步。按Actor次数对齐：B57比A63更早，所以上述“晚188步”不能被解释为稳定性改善。按Actor次数比时B已经额外做过200次Critic更新；两种横轴都不能把所有机制压成一个变量。

A的22/50/63次后三维策略取下一次Actor前的已保存probe，Actor未变，但Critic已经再更新，因此该probe梯度不与B更新后的梯度直接配对；A100次取5200 raw probe。[两种对齐数据](actor_count_alignment.json)、[逐更新指标](B/updates.json)、[完整固定动作数组](B/fixed_policy_actions.npz)。

![Actor更新机制](actor_mechanism.png)

## 3. Critic排序与真实Full有限时域回报

复用D2的12个真实状态/重建动作前缀，世界距离/高度分组为near3、far4、above5、below4，分组重叠。同一状态首个动作固定D2原A6k XY，只扫az=[−1,−.5,0,.5,1]→世界vz=[−.3,−.15,0,.15,.3]m/s；XY仅固定**一个上层间隔**，后续控制可自由改变三轴，避免永久固定XY越界造成误判。

共同后续控制为冻结的原5k Actor；各自后续控制为对应checkpoint的Actor。B5200与原5k Actor完全相同，共同/各自返回同一物理轨迹；A6k各自后续的60条分支复用已校验的D2 fresh-reset产物。新增240条Full分支，每条均fresh reset＋原真实动作前缀；12状态观测重构误差0，同一首动作重复一步结果完全相同；不使用PyBullet save/restore。

每条90上层步同时提取30步/90步即2s/6s折扣回报，raw reward×.1、gamma.99，与Q量纲一致。早停后不继续积分，未终止不加Critic bootstrap，避免循环以Critic自己证明排序。每状态10个动作对，回报差<1e−6或Q差<1e−7的tie剔除，逐状态Spearman和top-action匹配另存。下面仅是有限诊断样本的一致率，动作对相关，不能按120个独立样本作总体显著性论断。

| checkpoint | 后续控制 | 时域 | Q1一致率 | Q2一致率 | min-Q一致率 |
| --- | --- | --- | --- | --- | --- |
{rank_table}

B先行完成时，共同6s的Q1为30/102非tie动作对一致，初始为56/102；A5200与B5200排序统计完全相同，虽然Q数值不同。B自己的Actor尚未变化，依然没有建立可靠的诊断排序；但共享流本身包含A更新后采集的失稳经验，所以没有排除A行为数据质量的作用。共同2s的min-Q有39.2%→45.1%的局部提升，而6s三种Critic通道均下降，不能声称所有指标都恶化。B6k各自6s整体Q1/min-Q为44.9%/41.1%，与A6k各自46.8%/46.8%相比没有明确改善；各自控制的真实回报不同，不能用A回报替代B真值。

| checkpoint | 世界分组 | 状态数 | 共同2s Q1一致率 | 共同6s Q1一致率 |
| --- | --- | --- | --- | --- |
{group_table}

目标下方组有局部改善，近目标、目标上方及全体6s没有总体改善，不能声称每个状态/分组都恶化。共同后续控制在B5200就是自身策略，是最直接的先行Critic诊断；B5325/B6000对共同π5k仅是跨组参照，不是其自身无限时域Q真值。部分代表性状态采自原A后续5k→6k，对5200而言尚未进入Replay；因此该集合也含分布外诊断。当前Critic学的是平滑target策略，实测为确定性有限时域，二者仍有差异；90步回报不能当无限时域Q真值。

![12状态垂直排序](vertical_value_ranks.png)

完整Q1/Q2/min扫描、Actor实际选择动作及其价值、每候选真实轨迹见[paired_values.json](paired_values.json)；逐状态排序及tie为[ranking_pairs.json](ranking_pairs.json)，分组为[ranking_summary.json](ranking_summary.json)，重建证明为[physical_proof.json](physical_proof.json)。

## 4. 同图Full导航评估

validation20固定Stage0地图；L0使用相同QA剩余路径约.8m起点，原始任务固定[-8,0,1]→[8,0,1]。所有reset初速度0，成功严格<.2m、引导<1.2m、40s和终止优先顺序原样。所有候选策略重新评估；B5200从hash相同A5000复用。原A5000/A6000全部逐图L0/16m结局与D1既有数据一致。

0.3/.5到达、最小终距、参考跟踪误差在本轮均按120Hz下层记录；不改变正式成功条件。每回合保存upper动作/位置/速度和lower位置/速度/Reference/终距/接触，记录真实净空和入区时间/速度。

| checkpoint | 任务 | .2成功 | .3到达 | .5到达 | 碰撞 | XY越界 | 高度越界 | timeout | 平均最小终距m | 跟踪误差均值m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{eval_table}

相对初始A5k：B5325/B5600/B6000均失去原11个成功图，新增成功0，保持失败9；同图二项discordance检验p≈.000977，描述的是这20图上的退化，不是四seed泛化。相对A6k：B6k成功同为0，不能宣称成功率改善；L0失败组成从18XY＋2碰撞变为17XY＋3碰撞，.3/.5入区还从3/5降为2/4。

原始16m：B5325/B5600的19图XY越界，是负XY偏置导致倒退；B6k为15高度越界＋5碰撞，平均最小终距约13.113m，A6k约11.940m。高度失败数量较少伴随碰撞更多和更早终止，不能据此声称更安全。无策略在16m成功或进入1.2m。跟踪误差为厘米量级且动作对应实际运动，未观察到能解释崩溃的下层失控。

[逐图配对变化](paired_validation_changes.json)、[导航指标](validation_summary.json)、[完整逐回合记录](validation.json)、[原始轨迹](validation_traces)。固定5k的55%只是未训练Actor恰在这些近目标图上的能力；B5200保持55%是冻结恒等，不是学习成功。

## 5. 根因证据与后续决策

**已证实：** 共享数据/noise/优化超参数下，前200次Actor不动没有改善本集合的共同控制6s排序；解除冻结前Q1的三轴梯度已明显偏向负XY/正Z。Actor实际损失梯度沿这些方向更新，第57次全部正饱和。第63次到6k没有恢复，L0原能力丢失。它不是B探索恰好采到坏轨迹，因为B在受控阶段没有采集自己的训练经验。

**较强解释：** 当前早期数据、奖励/表示及TD目标组合仍产生不可靠局部动作斜率，Actor快速放大该斜率；“Critic只需比Actor早学200次”不足。Critic是否长期需要更多覆盖、Actor步幅是否过大、target滞后作用仍未分离；结果不能简单选定“Critic太慢”或“Actor太激进”中的唯一答案。

**未验证：** 改Actor LR、梯度上限、延长预热、修正M0任务契约能否恢复学习；不同训练seed及真实在线分布反馈是否会改变结论。本轮不能证明critic lead对所有种子有害，也不建议机械增加到500/1000次。

本结果仅约束固定A行为数据上的先行机制。B若在线采集自己的经验，可能形成不同Replay及成功样本；这种反馈未测试，不能把离线负结果等同于所有在线方案必然无效。本轮仍按事前顺序不启动在线分支。

按事前规则，B未达到L0至少9/20且相对A6k净增至少5图、Actor63和6k饱和率<50%的能力保留条件。因此不补C，不进入独立在线5k→6k/8k；没有执行任何正式10k/20k续训。C未来用于有收益时分离“Critic提前学习”与“仅延迟联合更新”，当前没有收益可分离。[决策记录](decisions.json)。

下一步应优先审查早期Replay状态—动作覆盖、少数成功轨迹和终止target对Critic更新的贡献、输入尺度及M0契约。若后来能先证明局部价值排序有所改善而Actor仍迅速退化，才把D4 Actor LR 3e−4 vs1e−4作为独立机制对照。本轮不修改M0发现的目标截断、引导速度域、近目标罚分或gamma，不加入K1/行为克隆。

## 6. 产物、校验与复现

独立目录保留raw网络/Adam/targets、每更新动作数组、Q分布/梯度/TD误差、Replay来源、配置指纹、逐图结果、科学图PNG/PDF。完整来源{len(protected)}份文件SHA256前后未变，A没有重新训练；B新增Critic1000/Actor400，仅受控离线消融。3项测试通过：采样拒绝未来转移、实际A第一批target/critic loss相同并验证冻结与target节拍、Q排序tie处理。

入口：[运行脚本](../../scripts/run_ego20x10_d3.py)、[物理/导航评估](../../scripts/evaluate_ego20x10_d3.py)、[汇总脚本](../../scripts/summarize_ego20x10_d3.py)、[测试](../../tests/test_ego20x10_d3.py)。先`run_ego20x10_d3 train`，再`evaluate_ego20x10_d3 values/evaluation`，最后汇总。运行脚本首次要求新目录，可指定`--output`；本次原目录不可重复覆盖。C入口存在但未调用，在线入口本轮未实现/未运行。
'''
    (ROOT/'REPORT.zh.md').write_text(report)
    for name in ('train','values','eval','tests'):
        shutil.copyfile('/tmp/ego20x10_d3_'+name+'.log',ROOT/(name+'.log'))
    source_files=[Path('scripts/run_ego20x10_d3.py'),Path('scripts/evaluate_ego20x10_d3.py'),Path('scripts/summarize_ego20x10_d3.py'),Path('tests/test_ego20x10_d3.py')]
    write(ROOT/'provenance.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),source_5k_sha256=protocol['source_5k_sha256'],
                                    script_sha256={str(p):sha(p) for p in source_files},tests_passed=3,physical_branches=240,
                                    validation_maps=20,validation_physical_episodes=200,**guard))
    print('[D3] report complete; no C/online trigger')


if __name__=='__main__':main()
