"""Cross-map conflicts and equal-budget shuffled-latent diagnostic probes.

Only train/validation maps are used. These offline probes cannot demonstrate
navigation success, latent causality, or irreducible partial observability.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist

from models.encoder import Encoder
from scripts.evaluate_native_latent_actor import write_json
from scripts.freeze_logical_async import file_record
from high_level_planner.coordinate_contract import future_to_world, rotate_z


def block_standardize(values, train_mask, block_sizes):
    values=np.asarray(values,dtype=np.float64)
    if sum(block_sizes)!=values.shape[1]:
        raise ValueError("Block sizes must span all feature dimensions")
    parts=[]
    offset=0
    for size in block_sizes:
        block=values[:,offset:offset+size]
        mean=block[train_mask].mean(axis=0)
        std=np.maximum(block[train_mask].std(axis=0),1e-3)
        parts.append((block-mean)/std/np.sqrt(size*len(block_sizes)))
        offset+=size
    return np.concatenate(parts,axis=1)


def cross_map_neighbors(features, map_ids, query_indices, reference_indices, k=5):
    """At most ONE representative from each other map in each neighbor set."""
    features=np.asarray(features)
    map_ids=np.asarray(map_ids)
    reference_indices=np.asarray(reference_indices)
    reference_maps=np.unique(map_ids[reference_indices])
    out_indices,out_distances=[],[]
    for start in range(0,len(query_indices),256):
        queries=np.asarray(query_indices[start:start+256])
        distances=cdist(features[queries],features[reference_indices])
        best_indices,best_distances=[],[]
        for identity in reference_maps:
            columns=np.flatnonzero(map_ids[reference_indices]==identity)
            nearest=np.argmin(distances[:,columns],axis=1)
            idx=columns[nearest]
            dist=distances[np.arange(len(queries)),idx]
            dist=np.where(map_ids[queries]==identity,np.inf,dist)
            best_indices.append(reference_indices[idx])
            best_distances.append(dist)
        if not best_indices:
            raise ValueError("No reference maps available")
        best_indices=np.stack(best_indices,axis=1)
        best_distances=np.stack(best_distances,axis=1)
        order=np.argsort(best_distances,axis=1)[:,:min(k,len(reference_maps))]
        out_indices.append(np.take_along_axis(best_indices,order,axis=1))
        out_distances.append(np.take_along_axis(best_distances,order,axis=1))
    return np.concatenate(out_indices),np.concatenate(out_distances)


def turn_labels(target, obstacle):
    direction=np.asarray(target)[:,0,:2]
    obstacle=np.asarray(obstacle)
    cross=obstacle[:,0]*direction[:,1]-obstacle[:,1]*direction[:,0]
    denom=np.linalg.norm(obstacle,axis=1)*np.linalg.norm(direction,axis=1)
    sine=np.divide(cross,denom,out=np.zeros_like(cross),where=denom>1e-8)
    eligible=(np.linalg.norm(obstacle,axis=1)<1.75)&(obstacle[:,0]>0)&(np.linalg.norm(direction,axis=1)>.015)
    return np.where(eligible & (np.abs(sine)>.15),np.sign(sine),0).astype(int)


def conflict_report(data,features,blocks):
    train=data["split"]=="train"
    # Bound query count per map and avoid giving long episodes more weight.
    queries=np.concatenate([np.flatnonzero(data["map_id"]==identity)[::4] for identity in np.unique(data["map_id"][~train])])
    refs=np.flatnonzero(train)[::2]
    scaled=block_standardize(features,train,blocks)
    indices,distances=cross_map_neighbors(scaled,data["map_id"],queries,refs)
    a=data["reference_target"][queries,0,:2]
    b=data["reference_target"][indices,0,:2]
    norm=np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=2)
    cosine=np.divide(np.sum(a[:,None,:]*b,axis=2),norm,out=np.ones_like(norm),where=norm>1e-8)
    angles=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    moving=(np.linalg.norm(a,axis=1)[:,None]>.015)&(np.linalg.norm(b,axis=2)>.015)
    labels=data.get("side_labels")
    if labels is None:
        labels=turn_labels(data["reference_target"],data["obstacle_local"])
    side_eligible=(labels[queries,None]!=0)&(labels[indices]!=0)
    finite=np.isfinite(distances)
    bins=[]
    for radius in (.1,.25,.5,1.):
        mask=finite&(distances<=radius)&moving
        side=mask&side_eligible
        bins.append({"standardized_radius":radius,"moving_pairs":int(mask.sum()),
                     "opposite_heading_rate":float((angles[mask]>90).mean()) if mask.any() else None,
                     "side_eligible_pairs":int(side.sum()),"opposite_side_rate":float((labels[queries,None]!=labels[indices])[side].mean()) if side.any() else None})
    pairs=[]
    order=np.argsort(np.where(finite & moving & (angles>90),distances,np.inf),axis=None)
    for flat in order[:20]:
        row,col=np.unravel_index(flat,distances.shape)
        if not finite[row,col] or not moving[row,col] or angles[row,col]<=90:
            continue
        i,j=int(queries[row]),int(indices[row,col])
        first_clearance,second_clearance=data["future_minimum_clearance"][[i,j]]
        clearance_difference=abs(first_clearance-second_clearance) if np.isfinite([first_clearance,second_clearance]).all() else np.nan
        pairs.append({"query_map":data["map_id"][i],"neighbor_map":data["map_id"][j],"query_row":i,"neighbor_row":j,
                      "distance":distances[row,col],"angle_degrees":angles[row,col],"turn_labels":[int(labels[i]),int(labels[j])],
                      "clearance_difference":float(clearance_difference) if np.isfinite(clearance_difference) else None,
                      "success_labels":[bool(data["success"][i]),bool(data["success"][j])]})
    return {"query_samples":len(queries),"train_reference_samples":len(refs),"query_maps":len(np.unique(data["map_id"][queries])),
            "k":indices.shape[1],"distinct_neighbor_maps":True,"no_same_map_neighbors":bool(np.all(data["map_id"][queries,None]!=data["map_id"][indices])),
            "nearest_distance_median":float(np.median(distances[:,0])),"radius_analysis":bins,"closest_opposite_pairs":pairs,
            "interpretation":"Finite-data near-neighbor conflicts; not proof of identical observations or irreducible aliasing. Sparse ray resampling is approximate."}


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        # All five conditions have identical trainable parameter count. The
        # first 11 slots are shared local goal/ranges; remaining 263 hold raw,
        # latent or zeros. No condition gets a larger hidden network.
        self.net=nn.Sequential(nn.Linear(274,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,30))
    def forward(self,x):
        return self.net(x)


def build_probe_features(data,seed):
    x=data["planner_input"].astype(np.float32)
    features=np.zeros((len(x),274),dtype=np.float32)
    features[:,:11]=x[:,16:]
    conditions={"P0_goal_ranges":features.copy()}
    features[:,11:27]=x[:,:16]
    conditions["P1_task_latent"]=features.copy()
    shuffled=features.copy()
    rng=np.random.default_rng(seed)
    for split in ("train","validation"):
        rows=np.flatnonzero(data["split"]==split)
        shuffled[rows,11:27]=x[rng.permutation(rows),:16]
    conditions["P2_shuffled_latent"]=shuffled
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+20000)
        encoder=Encoder(263,16).eval().requires_grad_(False)
        with torch.inference_mode():
            random_latent=encoder(torch.tensor(data["raw_state"],dtype=torch.float32)).numpy()
    random_features=features.copy()
    random_features[:,11:27]=random_latent
    conditions["P3_random_frozen_latent"]=random_features
    raw_features=features.copy()
    raw_features[:,11:]=data["raw_state"]
    conditions["P4_raw_observation"]=raw_features
    return conditions


def train_probes(data,epochs,seeds):
    train=np.flatnonzero(data["split"]=="train")
    val=np.flatnonzero(data["split"]=="validation")
    targets=data["reference_target"].reshape(-1,30).astype(np.float32)
    mean=targets[train].mean(axis=0)
    std=np.maximum(targets[train].std(axis=0),.01)
    truth=torch.from_numpy((targets-mean)/std)
    results={}
    saved=None
    batch_size=256
    for seed in seeds:
        features=build_probe_features(data,seed)
        torch.manual_seed(seed)
        initial=copy.deepcopy(Probe().state_dict())
        for name,values in features.items():
            mu=values[train].mean(axis=0)
            sigma=np.maximum(values[train].std(axis=0),1e-3)
            x=torch.from_numpy((values-mu)/sigma)
            model=Probe()
            model.load_state_dict(initial)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
            rng=np.random.default_rng(seed)
            for _ in range(epochs):
                for offset in range(0,len(train),batch_size):
                    # Permutation prepared once per epoch below, same across conditions.
                    if offset==0:
                        order=rng.permutation(train)
                    idx=order[offset:offset+batch_size]
                    pred=model(x[idx]).reshape(-1,5,6)
                    expected=truth[idx].reshape(-1,5,6)
                    loss=(pred[:,:,:3]-expected[:,:,:3]).square().mean()+.5*(pred[:,:,3:]-expected[:,:,3:]).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.inference_mode():
                prediction=model(x[val]).numpy()*std+mean
            error=(prediction-targets[val]).reshape(-1,5,6)
            per_map={}
            for identity in np.unique(data["map_id"][val]):
                group=error[data["map_id"][val]==identity]
                per_map[str(identity)]={"position_mse":float(np.mean(group[:,:,:3]**2)),"velocity_mse":float(np.mean(group[:,:,3:]**2))}
            item={"seed":seed,"epochs":epochs,"gradient_steps":epochs*int(np.ceil(len(train)/batch_size)),
                  "train_samples":len(train),"validation_samples":len(val),"parameter_count":sum(p.numel() for p in model.parameters()),
                  "map_macro_position_rmse":float(np.sqrt(np.mean([v["position_mse"] for v in per_map.values()]))),
                  "map_macro_velocity_rmse":float(np.sqrt(np.mean([v["velocity_mse"] for v in per_map.values()]))),"per_map":per_map}
            results.setdefault(name,[]).append(item)
            print(f"probe {name} seed={seed}: pRMSE={item['map_macro_position_rmse']:.4f}, vRMSE={item['map_macro_velocity_rmse']:.4f}",flush=True)
            if name=="P1_task_latent" and seed==seeds[0]:
                saved={"model":model,"input_mean":mu,"input_std":sigma,"target_mean":mean,"target_std":std,"features":values}
    return results,saved


def paired_interval(results,baseline):
    ids=sorted(results["P1_task_latent"][0]["per_map"])
    def errors(name):
        return np.mean([[np.sqrt(run["per_map"][m]["position_mse"]) for m in ids] for run in results[name]],axis=0)
    differences=errors(baseline)-errors("P1_task_latent")
    rng=np.random.default_rng(404)
    boot=np.mean(differences[rng.integers(0,len(ids),size=(2000,len(ids)))],axis=1)
    return {"comparison":baseline+" minus P1", "positive_favors_task_latent":True,
            "mean_map_position_rmse_improvement":float(differences.mean()),"paired_map_bootstrap_95ci":np.quantile(boot,[.025,.975]).tolist(),
            "maps":len(ids),"training_seeds_averaged":len(results[baseline]),"unit":"m"}


def prediction_clearance(sequence,positions,yaws,map_ids,maps):
    result=[]
    for target,position,yaw,identity in zip(sequence,positions,yaws,map_ids):
        world,_=future_to_world(target,position,yaw)
        points=np.vstack([position,world])
        interpolated=np.concatenate([a+np.linspace(0,1,11)[:,None]*(b-a) for a,b in zip(points[:-1],points[1:])])
        trees=maps[str(identity)]["trees"]
        # Diagnostic planar tube uses CF2X radius, not merely a point path.
        clearance=min((np.min(np.linalg.norm(interpolated[:,:2]-t["xy"],axis=1))-t["radius"]-.06 for t in trees),default=np.inf)
        result.append(clearance)
    return np.asarray(result)


def sensitivity(data,saved,maps):
    val=np.flatnonzero(data["split"]=="validation")[::8]
    base=saved["features"][val].copy()
    def predict(x):
        with torch.inference_mode():
            value=saved["model"](torch.tensor((x-saved["input_mean"])/saved["input_std"])).numpy()
        return (value*saved["target_std"]+saved["target_mean"]).reshape(-1,5,6)
    reference=predict(base)
    reference_clearance=prediction_clearance(reference,data["position"][val],data["yaw"][val],data["map_id"][val],maps)
    modifications={}
    for mode in ("mask_latent","shuffle_latent","mask_ranges_as_unknown_zero","ranges_all_clear","front_range_near","rotate_goal_90deg"):
        changed=base.copy()
        if mode=="mask_latent": changed[:,11:27]=0
        elif mode=="shuffle_latent": changed[:,11:27]=base[np.random.default_rng(7).permutation(len(base)),11:27]
        elif mode=="mask_ranges_as_unknown_zero": changed[:,3:11]=0
        elif mode=="ranges_all_clear": changed[:,3:11]=1
        elif mode=="front_range_near": changed[:,3]=.05
        else: changed[:,:2]=np.column_stack([-base[:,1],base[:,0]])
        prediction=predict(changed)
        clearance=prediction_clearance(prediction,data["position"][val],data["yaw"][val],data["map_id"][val],maps)
        finite=np.isfinite(clearance)&np.isfinite(reference_clearance)
        a,b=reference[:,0,:2],prediction[:,0,:2]
        norm=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        angles=np.degrees(np.arccos(np.clip(np.divide(np.sum(a*b,axis=1),norm,out=np.ones_like(norm),where=norm>1e-8),-1,1)))
        old_labels,new_labels=turn_labels(reference,data["obstacle_local"][val]),turn_labels(prediction,data["obstacle_local"][val])
        eligible=(old_labels!=0)&(new_labels!=0)
        modifications[mode]={"mean_trajectory_position_change_m":float(np.linalg.norm(prediction[:,:,:3]-reference[:,:,:3],axis=2).mean()),
                             "mean_initial_heading_change_deg":float(angles.mean()),"mean_clearance_change_m":float((clearance[finite]-reference_clearance[finite]).mean()) if finite.any() else None,
                             "side_flip_fraction":float((old_labels[eligible]!=new_labels[eligible]).mean()) if eligible.any() else None}
    return {"samples":len(val),"interventions":modifications,"counterfactual_warning":"Input ablations may be physically inconsistent or out-of-distribution; response magnitude alone does not prove safe obstacle avoidance.",
            "clearance_definition":"piecewise-linear raw predicted position tube, CF2X radius 0.06m, constant-altitude obstacles; filtering/PID not executed in probes"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="data/planner_diagnostics/map_isolated_v1")
    parser.add_argument("--output",default="analysis_outputs/planner_observability_v1")
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--conflicts-only",action="store_true",help="Refresh conflict statistics without retraining existing probes.")
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=args.conflicts_only)
    torch.set_num_threads(1)
    with np.load(Path(args.dataset)/"samples.npz",allow_pickle=False) as archive:
        data={key:archive[key] for key in archive.files}
    if set(data["split"])-{"train","validation"}:
        raise ValueError("Diagnostics cannot consume final test samples")
    if set(data["map_id"][data["split"]=="train"])&set(data["map_id"][data["split"]=="validation"]):
        raise ValueError("Training/validation map leakage")
    x=data["planner_input"]
    legacy_x=np.concatenate([x[:,:16],data["raw_state"][:,-11:]],axis=1)
    legacy_data=dict(data)
    # Side eligibility is defined in ego heading coordinates and is invariant
    # to rotating target/vector into world coordinates for legacy angle tests.
    legacy_data["side_labels"]=turn_labels(data["reference_target"],data["obstacle_local"])
    legacy_data["reference_target"]=np.asarray([np.concatenate([rotate_z(y[:,:3],yaw),rotate_z(y[:,3:],yaw)],axis=1) for y,yaw in zip(data["reference_target"],data["yaw"])])
    legacy_data["obstacle_local"]=np.asarray([rotate_z(np.r_[obstacle,0.],yaw)[:2] for obstacle,yaw in zip(data["obstacle_local"],data["yaw"])])
    history_features=np.concatenate([x[:,:19],data["history"]],axis=1)
    conflicts={"legacy_world_27d":conflict_report(legacy_data,legacy_x,[16,3,8]),
               "local_27d":conflict_report(data,x,[16,3,8]),
               "goal_ranges_11d":conflict_report(data,x[:,16:],[3,8]),
               "same_context_plus_history_83d":conflict_report(data,history_features,[16,3,64])}
    write_json(root/"conflicts.json",conflicts)
    if args.conflicts_only:
        return
    results,saved=train_probes(data,args.epochs,args.seeds)
    maps={r["map_id"]:r for r in json.loads((Path(args.dataset)/"maps.json").read_text())}
    report={"dataset":file_record(Path(args.dataset)/"samples.npz"),"coordinate_contract":"planner_local_coordinates_v1",
            "purpose":"offline_diagnostic_only_not_navigation_training", "test_maps_used":False,
            "probe_input_contract":"New local27 context, not the frozen legacy-world27 Planner. Legacy and local nearest-neighbor conflicts are reported separately.",
            "budget":{"epochs":args.epochs,"seeds":args.seeds,"batch_size":256,"optimizer":"Adam lr=1e-3","architecture":"274 zero-padded input -> 128 -> 128 -> 30", "selection":"fixed final epoch, no early stopping or hyperparameter search"},
            "results":results,"paired_comparisons":[paired_interval(results,name) for name in ("P0_goal_ranges","P2_shuffled_latent","P3_random_frozen_latent","P4_raw_observation")],
            "sensitivity":sensitivity(data,saved,maps),
            "limitations":["Global grid teacher is privileged and not a valid native Actor success baseline.","Only 36 train/18 validation maps by default; no final test or closed-loop navigation claim.","Approximate ray rebinning can itself introduce aliasing.","Low error or ablation sensitivity cannot establish latent navigation advantage.","A small mean tracking error in earlier runs does not exclude localized PID or reference-filter failures."]}
    write_json(root/"report.json",report)


if __name__=="__main__":
    main()
