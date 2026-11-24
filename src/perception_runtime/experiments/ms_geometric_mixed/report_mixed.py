"""Summarize mixed-course and prior hard-course 40k results with matched cases."""
import csv
import json
from pathlib import Path
import sys
import time
import zipfile
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODES=("original","directional")
COURSES=("hard","mixed")
LABELS={("hard","original"):"Original A4 / hard",("hard","directional"):"Directional / hard",
        ("mixed","original"):"Original A4 / mixed",("mixed","directional"):"Directional / mixed"}
ZH={("hard","original"):"原课程／原 A4",("hard","directional"):"原课程／方向增强版",
    ("mixed","original"):"混合课程／原 A4",("mixed","directional"):"混合课程／方向增强版"}


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wait_for(mixed):
    until=time.monotonic()+3*3600
    while time.monotonic()<until:
        state=read(mixed/"status.json")
        if state["phase"]=="failed":
            raise RuntimeError(state)
        if state["phase"]=="evaluation_complete":
            return state
        time.sleep(15)
    raise TimeoutError("mixed experiment not complete")


def comparison(rows):
    groups={}
    for course in COURSES:
        for mode in MODES:
            relevant=[r for r in rows if r["course"]==course and r["mode"]==mode]
            group=[sum(r["reason"]=="success" for r in relevant if r["stage"]==stage) for stage in range(4)]
            collisions=[sum(r["collision_observed"] for r in relevant if r["stage"]==stage) for stage in range(4)]
            goals=[float(np.median([r["final_goal_distance_m"] for r in relevant if r["stage"]==stage])) for stage in range(4)]
            groups[course,mode]=dict(success=group,collisions=collisions,goal_distance_median=goals,
                                      outcomes={str(s):dict(Counter(r["reason"] for r in relevant if r["stage"]==s))
                                                for s in range(4)})
    return groups


def make_figure(dest,forest,paired,exposure):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    baseline=np.arange(4)
    for i,((course,mode),s) in enumerate(forest.items()):
        offset=(i-1.5)*.19
        axes[0,0].bar(baseline+offset,100*np.asarray(s["success"])/20,.18,label=LABELS[course,mode])
        axes[0,1].bar(baseline+offset,100*np.asarray(s["collisions"])/20,.18,label=LABELS[course,mode])
        p=[r for r in paired if r["course"]==course and r["mode"]==mode]
        axes[1,0].plot([r["case"] for r in p],[r["delta_y_mps"] for r in p],marker=".",label=LABELS[course,mode])
    for i,mode in enumerate(MODES):
        x=exposure[mode]["final_metrics"]["exposure"]
        axes[1,1].bar(baseline+(i-.5)*.36,100*np.asarray(x)/40000,.34,label=LABELS["mixed",mode])
    axes[1,0].axhline(.08,color="grey",ls="--",label="Contrast threshold")
    axes[1,0].axhline(0,color="grey",lw=.6)
    labels=[("Goal reached","Validation obstacle stage","Episodes (%)"),
            ("Collision flag observed","Validation obstacle stage","Episodes (%)"),
            ("Mirrored opening: initial lateral response","Paired scenario","Lateral command difference (m/s)"),
            ("Actual mixed-course exposure","Training obstacle stage","Transitions (%)")]
    for i,(ax,(title,x,y)) in enumerate(zip(axes.ravel(),labels)):
        ax.set(title=title,xlabel=x,ylabel=y)
        if i!=2:
            ax.set_xticks(range(4))
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    fig.savefig(dest/"混合课程_40k对照.png",dpi=175)
    plt.close(fig)


def main():
    hard,mixed,out=map(Path,sys.argv[1:4])
    done=wait_for(mixed)
    directories={"hard":hard,"mixed":mixed}
    reports={k:read(p/"evaluation_40000/results.json") for k,p in directories.items()}
    protocols={k:read(p/"evaluation_40000/protocol.json") for k,p in directories.items()}
    assert protocols["hard"]==protocols["mixed"]
    common=done["checks"]["common_episode_start_prefix"]
    sample=[]
    pairs=[]
    for course,data in reports.items():
        assert len(data["forest"])==160 and len(data["probe"])==48
        for item in data["forest"]:
            init=item["init"]
            sample.append(dict(course=course,**{k:v for k,v in item.items() if k!="init"}))
        pairs.extend(dict(course=course,**item) for item in data["actor_paired"])
    ref=sorted([(r["mode"],r["stage"],r["case"],r["init"]) for r in reports["hard"]["forest"]],
               key=lambda r:r[:3])
    new=sorted([(r["mode"],r["stage"],r["case"],r["init"]) for r in reports["mixed"]["forest"]],
               key=lambda r:r[:3])
    assert ref==new
    forest=comparison(sample)
    detail=done["checks"]["details"]
    for mode in MODES:
        r=detail[mode]["final_metrics"]
        assert r["updates"]==35001 and r["actor_updates"]==17500
        assert sum(r["exposure"])==40000
        assert sum(r["sampled_main"]+r["sampled_success"])==256*35001
    dest=ROOT/"MSGP_RESEARCH_REPORT/e07_e10/mixed_40k_comparison"
    dest.mkdir(exist_ok=True)
    write_csv(dest/"forest_40k_episodes.csv",sample)
    write_csv(dest/"opening_40k_pairs.csv",pairs)
    make_figure(dest,forest,pairs,detail)
    stats=[]
    for mode in MODES:
        item=detail[mode]
        m=item["final_metrics"]
        stats.append(f'| {ZH["mixed",mode]} | {item["episode_starts"]} | {m["exposure"]} | {m["sampled_main"]} | {m["sampled_success"]} |')
    stage=[]
    for s in range(4):
        vals=[f'{forest[key]["success"][s]}/20' for key in forest]
        counts=[str(forest[key]["collisions"][s]) for key in forest]
        stage.append('| Stage'+str(s)+' | '+' | '.join(vals+counts)+' |')
    probe=[]
    for c in COURSES:
        for m in MODES:
            d=reports[c]["summary"][m]["probe"]
            med=d["median_delta_y_mps"]
            probe.append(f'| {ZH[c,m]} | {d["eligible_only_far_varies"]}/12 | {d["contrasting_pairs"]}/12 | {d["correct_signed_pairs"]}/12 | {med:.3f} | {d["early_turn"]}/24 | {d["passed_gate"]}/24 | {d["reasons"].get("success",0)}/24 | {d["reasons"].get("collision",0)}/24 |')
    gate=read(mixed/"gate_6000.json")
    assert read(mixed/"resume_verification.json")["exact"]
    md=f"""# 混合障碍课程 40k 对照

原 A4、方向增强版分别从头用 seed 43 和 CUDA 训练至 40k。按每 20 个回合起点随机打散 10/5/3/2 个 Stage0–3 的固定配额。两组共享按回合序号确定的障碍阶段序列。沿用原起点课程、奖励、控制和 TD3 配置。共同前缀的 {common} 个回合核对了地图索引、几何哈希及障碍阶段。两组策略会改变回合长度、起点课程晋升及起点抽样轨迹。

两组 6k 更新检查、checkpoint 哈希和恢复通过，再继续至 40k。每组 40k 时有 35,001 次 Critic 更新和 17,500 次 Actor 更新。短程检查中 64→128 的恢复与连续运行逐项相同。40k 时实际交互和 Replay 计数如下，均按 Stage0、1、2、3 顺序。

| 组别 | 回合起点数 | 真实交互步 | 主 Replay 抽样数 | 成功 Replay 抽样数 |
|---|---|---|---|---|
{chr(10).join(stats)}

Replay 抽样计数允许同一记录重复抽到。原课程两组的 40k 全在 Stage0。本轮训练与各组独立的策略轨迹仍只有一个种子。

## 固定森林验证

每模型每阶段 20 个原始起点。四个模型逐回合配对地图和起始物理状态。回合到达遵循原环境的成功判定。碰撞列报告上层步观测到的碰撞标志。

| 阶段 | 原课／原 A4 到达 | 原课／方向增强到达 | 混合／原 A4 到达 | 混合／方向增强到达 | 原课／原 A4 碰撞 | 原课／方向增强碰撞 | 混合／原 A4 碰撞 | 混合／方向增强碰撞 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(stage)}

## 策略利用诊断

相同 12 对镜像开口。每组 24 个闭环回合。侧向对比达标表示开口从负 y 换到正 y 后初始侧向命令增加至少 0.08 m/s。双侧均对要求各自在正确方向至少 0.04 m/s。提前转向用墙在 1 m 外、连续 3 个采样周期朝开口发出至少 0.08 m/s 的指令定义。仅提前向某一侧运动不能说明左右判断正确。

| 组别 | 有效配对 | 命令差达标 | 双侧均对 | 命令差中位数 m/s | 提前转向 | 过门 | 到达目标 | 碰撞终止 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(probe)}

评估沿用上一轮的森林场景与开口干预。混合课程影响训练分布，比较原课程与混合课程只能得出这一种子和短预算下的结果。240k 未启动，正式长训由用户据结果选择。

## 证据

混合课程原始文件位于 {mixed}。原课程数据位于 {hard}。所有冻结场景输入对照、逐例结果与配对动作统计见压缩包。原始轨迹也保留在各运行的 evaluation_40000/aperture_traces。
"""
    name="混合课程_40k报告.md"
    (dest/name).write_text(md)
    summary=dict(mixed_schedule_prefix=common,mixed_training=detail,
                 forest={f"{c}_{m}":v for (c,m),v in forest.items()},
                 opening={c:{m:r["summary"][m]["probe"] for m in MODES} for c,r in reports.items()})
    (dest/"summary.json").write_text(json.dumps(summary,indent=2))
    out.mkdir(parents=True,exist_ok=True)
    for n in (name,"混合课程_40k对照.png"):
        (out/n).write_bytes((dest/n).read_bytes())
    with zipfile.ZipFile(out/"混合课程_40k证据包.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(dest.iterdir()):
            if p.is_file():
                archive.write(p,p.name)
        for c,p in directories.items():
            for n in ("protocol.json","results.json"):
                archive.write(p/"evaluation_40000"/n,f"{c}/evaluation/{n}")
        for mode in MODES:
            for n in ("metadata.json","events.jsonl","metrics.jsonl","episodes.jsonl","episode_starts.jsonl"):
                archive.write(mixed/mode/n,f"mixed/training/{mode}/{n}")
        for n in ("mixed_protocol.json","status.json","resume_verification.json","gate_6000.json"):
            archive.write(mixed/n,f"mixed/training/{n}")
        for n in ("mixed_course.py","train_mixed.py","supervise_mixed.py","report_mixed.py"):
            archive.write(HERE/n,f"scripts/{n}")
    with zipfile.ZipFile(out/"混合课程_40k证据包.zip") as archive:
        assert archive.testzip() is None
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
