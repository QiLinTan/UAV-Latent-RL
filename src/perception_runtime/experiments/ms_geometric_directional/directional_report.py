"""Export training and strategy-use results after the queued 40k evaluation."""
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
MODES={"original":"原 A4","directional":"方向增强版"}


def lines(path):
    return [json.loads(r) for r in path.read_text().splitlines()]


def write_csv(path,rows):
    with path.open("w",newline="") as s:
        writer=csv.DictWriter(s,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wait_for(evaldir):
    deadline=time.monotonic()+3*3600
    while time.monotonic()<deadline:
        if (evaldir/"error.json").is_file():
            raise RuntimeError((evaldir/"error.json").read_text())
        source=evaldir/"results.json"
        if source.is_file():
            try:
                return json.loads(source.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(15)
    raise TimeoutError("evaluation results were not generated")


def create_plots(run,dest,result):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    validation=[]
    traces=[]
    for i,(mode,style) in enumerate(zip(MODES,("Original A4","Directional"))):
        source=run/mode
        updates=lines(source/"updates.jsonl")
        metrics=lines(source/"metrics.jsonl")
        episodes=lines(source/"episodes.jsonl")
        xs=[x["step"] for x in episodes]
        ys=[x["reward"] for x in episodes]
        w=min(len(ys),15)
        if w:
            axes[0,0].plot(xs[w-1:],np.convolve(ys,np.ones(w)/w,mode="valid"),label=style)
        slices=[updates[j:j+1000] for j in range(0,len(updates),1000)]
        axes[0,1].plot([s[-1]["step"] for s in slices],[np.mean([r["critic_loss"] for r in s]) for s in slices],label=style)
        axes[1,0].plot([r["step"] for r in metrics],[r["exposure"][0] for r in metrics],label=style)
        stagewise=[]
        for stage,v in result["summary"][mode]["forest"].items():
            stats=dict(mode=mode,stage=stage,n=v["n"],success=v["reason"].get("success",0),
                      collision=v["reason"].get("collision",0),timeout=v["reason"].get("timeout",0),
                      bounds=sum(v["reason"].get(k,0) for k in ("xy_bound","height_bound")),
                      median_final_goal_m=v["goal_distance_median"])
            validation.append(stats)
            stagewise.append(100*stats["success"]/stats["n"])
        axes[1,1].plot(range(4),stagewise,marker="o",label=style)
    labels=[("Training reward (15-episode mean)","Training steps","Episode reward"),
            ("Critic loss","Training steps","MSE"),
            ("Stage0 transitions","Training steps","Transitions"),
            ("Frozen 40k checkpoint","Obstacle stage","Success (%)")]
    for ax,(title,x,y) in zip(axes.ravel(),labels):
        ax.set(title=title,xlabel=x,ylabel=y)
        ax.grid(alpha=.25)
        ax.legend()
    axes[1,1].set_xticks(range(4))
    axes[1,1].set_ylim(-3,103)
    fig.savefig(dest/"40k_训练与闭环.png",dpi=175)
    plt.close(fig)
    write_csv(dest/"stage0_3.csv",validation)
    write_csv(dest/"aperture_action_pairs.csv",result["actor_paired"])
    pairs=result["actor_paired"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for mode,style in zip(MODES,("Original A4","Directional")):
        subset=[r for r in pairs if r["mode"]==mode]
        axes[0].plot([r["case"] for r in subset],[r["delta_y_mps"] for r in subset],marker=".",label=style)
    axes[0].axhline(.08,ls="--",color="grey",label="Directional contrast threshold")
    axes[0].axhline(0,color="grey",lw=.6)
    axes[0].set(title="Initial lateral command divergence",xlabel="Mirrored aperture pair",ylabel="v_y(+ opening) - v_y(- opening) / m/s")
    for mode,style in zip(MODES,("Original A4","Directional")):
        for side in (-1,1):
            p=run/"evaluation_40000/aperture_traces"/f"{mode}_00_{side:+d}.json"
            doc=json.loads(p.read_text())
            xy=np.array([x["position"][:2] for x in doc["trajectory"]])
            axes[1].plot(xy[:,0],xy[:,1],linestyle="--" if side<0 else "-",label=f"{style}, aperture {side:+d}")
    axes[1].axvline(1.65,lw=.8,color="grey",label="Wall x-coordinate")
    axes[1].set(title="Aperture pair 0 (fixed before analysis)",xlabel="x (m)",ylabel="y (m)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(dest/"40k_策略利用.png",dpi=175)
    plt.close(fig)
    return validation


def main():
    run=Path(sys.argv[1]).resolve()
    out=Path(sys.argv[2])
    evaldir=run/"evaluation_40000"
    result=wait_for(evaldir)
    destination=ROOT/"MSGP_RESEARCH_REPORT/e07_e10/40k_directional"
    destination.mkdir(exist_ok=True)
    forest=create_plots(run,destination,result)
    status=json.loads((run/"status.json").read_text())
    update_6k=json.loads((run/"gate_6000.json").read_text())
    gate=json.loads((run/"resume_verification.json").read_text())
    assert gate["exact"] and len(result["forest"])==160 and len(result["probe"])==48
    assert all(sum(x["exposure"])==40000 for x in status["checks"].values())
    exp_rows=[]
    for mode,title in MODES.items():
        last=status["checks"][mode]
        verify=update_6k[mode]
        exp_rows.append(f'| {title} | {verify["updates"]} / {verify["actor_updates"]} | {last["updates"]} / {last["actor_updates"]} | {last["exposure"]} |')
    f_rows=[]
    for stage in range(4):
        a,b=[r for r in forest if r["stage"]==str(stage) or r["stage"]==stage]
        f_rows.append(f'| Stage{stage} | {a["success"]}/20 | {b["success"]}/20 | {a["collision"]} / {b["collision"]} | {a["timeout"]} / {b["timeout"]} | {a["median_final_goal_m"]:.2f} / {b["median_final_goal_m"]:.2f} |')
    p_rows=[]
    for mode,title in MODES.items():
        p=result["summary"][mode]["probe"]
        median=p["median_delta_y_mps"]
        p_rows.append(f'| {title} | {p["eligible_only_far_varies"]}/12 | {p["contrasting_pairs"]} | {p["correct_signed_pairs"]} | {p["early_turn"]}/24 | {p["passed_gate"]}/24 | {median if median is None else round(median,4)} |')
    early_check=[]
    for m in MODES:
        ps=result["summary"][m]["probe"]
        early_check.append((ps["contrasting_pairs"],ps["passed_gate"]))
    report=f"""# 方向增强版与原 A4 的 40k 对照

## 训练与恢复

两组从头用 seed 43 在同一 RTX 4080 SUPER 上并行训练。沿用 RV1 任务、控制、奖励、硬门槛课程与 TD3 设置，使用 batch 256。两组的观测为 320 与 448 维，隐藏层均为 256 × 256。第一层参数量随输入维度变化。

训练版在 304 帧归档扫描上与离线编码逐元素一致。小预算检查中，连续训练到 128 步和从 64 步 checkpoint 恢复到 128 步得到完全一致的网络、优化器、Replay、仿真状态和随机状态。下表是正式 batch 256 的更新次数。两组都保存了 6k，并从该 checkpoint 恢复至 40k。

| 组别 | 6k Critic / Actor 更新 | 40k Critic / Actor 更新 | 训练交互 Stage0–3 |
|---|---:|---:|---|
{chr(10).join(exp_rows)}

训练期间实际抽样的障碍阶段分布可查两组 metrics.jsonl 的 sampled_main 和 sampled_success。全部训练权重与原始日志保存于 {run}。

## 40k 固定森林验证

每组每阶段 20 例，使用既有验证池的原起点，核对地图哈希、初始物理状态与目标。碰撞和超时列为原 A4／方向增强版。成功沿用任务环境的目标半径。

| 阶段 | 原 A4 到达 | 方向增强版到达 | 碰撞终止 | 超时 | 终点目标距离中位数 m |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f_rows)}

## 策略利用诊断（计划中的 E12）

用 12 对真实仿真的镜像开口。左右两侧使用同样的飞行初态和目标。每对核对状态和近场完全一致，统计远场有差异的对数。命令差检验定义为开口从负 y 换到正 y 后初始侧向参考增加至少 0.08 m/s。双侧正确进一步要求两侧分别朝开口方向达到 0.04 m/s。提前转向要求墙面在近场 1 m 之外，且连续三次命令朝开口方向达到 0.08 m/s。闭环过门按轨迹在门板后侧平面处穿过并留出横向机体余量判定。

| 组别 | 只改远场的有效配对 | 侧向命令差达标 | 双侧方向均对 | 提前转向 | 闭环过门 | 有效配对侧向命令差中位数 m/s |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(p_rows)}

这批配对是机制诊断，结果应与森林评估和课程暴露一起解读。开口传感编码有区别、策略命令改变量、闭环通过是三个不同指标。40k 初筛保留各阶段的失败原因，不据单种子短训报告长期性能。240k 未启动。

## 复现

本报告附阈值、初态配对结果、逐例统计、40k 评估协议和轨迹源索引。训练代码在 WSL 项目的 experiments/ms_geometric_directional/。同一验证池的场景供方法对照，正式独立测试应另外保留。
"""
    p=destination/"40k_训练与策略利用报告.md"
    p.write_text(report)
    (destination/"comparison.json").write_text(json.dumps(result,indent=2))
    out.mkdir(parents=True,exist_ok=True)
    for name in (p.name,"40k_训练与闭环.png","40k_策略利用.png"):
        (out/name).write_bytes((destination/name).read_bytes())
    with zipfile.ZipFile(out/"40k_对照证据包.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(destination.iterdir()):
            if path.is_file():
                z.write(path,path.name)
        for name in ("protocol.json","status.json"):
            z.write(evaldir/name,f"eval/{name}")
        for name in ("gate_6000.json","resume_verification.json","encoding_verification.json","status.json"):
            z.write(run/name,f"training/{name}")
        for mode in MODES:
            for name in ("metadata.json","events.jsonl","metrics.jsonl","episodes.jsonl"):
                z.write(run/mode/name,f"training/{mode}/{name}")
    with zipfile.ZipFile(out/"40k_对照证据包.zip") as z:
        assert z.testzip() is None
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":
    main()
