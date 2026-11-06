"""Reproducible geometry, frozen-feature probes and CPU cost; no RL rollout training."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import copy
import gc
import hashlib
import json
import platform
import resource
import sys
import numpy as np
import pybullet as p
import torch
from torch import nn
from envs.map_atlas.runtime import MapRuntime
from experiments.rv1_range_cnn.benchmark import spec_from
from experiments.rv1_range_cnn.lidar import Lidar, LidarConfig
from experiments.rv1_range_cnn.models import RangeEncoder
from experiments.direct_velocity_full_v1.task_factory import load_config
from .contracts import PerceptionSpec, make_contract, source_manifest, digest
from .features import canonical_image, radial_images, NearBypass, sector_indices
from .models import Encoder, Actor, Critic
from .replay import Replay, SuccessReplay
from .td3 import RangeTD3

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = [('rv1', PerceptionSpec(encoder='single')),
            ('rv1_B', PerceptionSpec(encoder='single', bypass=True)),
            ('rv2_A', PerceptionSpec()), ('rv2_B', PerceptionSpec(bypass=True)),
            ('joint_bands', PerceptionSpec(encoder='joint_bands')),
            ('single_450k', PerceptionSpec(encoder='single_large'))]

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def count(module):
    return sum(p.numel() for p in module.parameters())

def timing(call, repeat=10):
    call(); times=[]
    for _ in range(repeat):
        t=perf_counter();call();times.append((perf_counter()-t)*1000)
    return dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),samples_ms=times)

def macs(module, arg):
    values=[];hooks=[]
    def hook(layer, inputs, output):
        if isinstance(layer,nn.Conv2d):
            values.append(output.numel()*layer.in_channels//layer.groups*np.prod(layer.kernel_size))
        else:values.append(output.numel()*layer.in_features)
    for m in module.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():module(arg)
    for h in hooks:h.remove()
    return int(sum(values))

def collect(out, catalog):
    """All predefined A-D train/validation maps x fixed viewpoint grid; no test."""
    client=p.connect(p.DIRECT);images=[];records=[];geometry=[];examples={}
    sensor=Lidar(LidarConfig());directions=sensor.config.directions()
    # Viewpoints frozen independently of policy performance. Invalid poses logged, not replaced.
    positions=[(x,y,z) for x in (-3.2,-2.5) for y in (-.35,0.,.35) for z in (.8,1.2)]
    ids=sector_indices(directions,PerceptionSpec(bypass=True));bypass=NearBypass(directions)
    t0=perf_counter()
    try:
        for data in catalog['maps']:
            if data['split'] not in ('train','validation') or data['scenario']=='E':continue
            spec=spec_from(data);p.resetSimulation(physicsClientId=client)
            bodies=MapRuntime(spec).create_bodies(client_id=client)
            shape=p.createCollisionShape(p.GEOM_PLANE,physicsClientId=client)
            ground=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,physicsClientId=client)
            for pos in positions:
                # Static QA poses only. Intersecting bounding primitives would be invalid sensor origins.
                invalid=False
                for o in spec.obstacles:
                    rel=np.array(pos)+[0,0,.02]-np.array(o.position)
                    invalid |= bool(np.all(abs(rel)<=np.array(o.half_extents)+.08)) if o.kind=='box' else bool(np.linalg.norm(rel[:2])<o.radius+.08 and abs(rel[2])<o.height/2+.08)
                if invalid:
                    geometry.append(dict(map_id=spec.map_id,position=pos,skipped='sensor origin inside expanded primitive'));continue
                scan=sensor.scan(client_id=client,position=pos,quaternion_xyzw=[0,0,0,1],timestamp_s=0,obstacle_ids=bodies,ground_ids=[ground])
                obs=dict(state=np.zeros(18,np.float32),range_m=scan.range_m,hit_mask=scan.hit_mask,scanned_mask=scan.scanned_mask)
                image=canonical_image(obs);x=torch.from_numpy(image)[None]
                bands=radial_images(x);summary=bypass(x).reshape(24,2).numpy()
                d=image[0]*8;near=(image[1]>0)&(d<1.5)
                # Input-level guarantee is nearest per sector, explicitly not every hit identity.
                expected=np.array([max(.08,float(d[near&(ids==i)].min()))/1.5 if np.any(near&(ids==i)) else 1. for i in range(24)])
                nearest_ok=np.allclose(expected,summary[:,0],atol=1e-7)
                surface=bands[0,:,0].numpy().sum(0)
                hit_once=np.array_equal(surface,(image[1]>0).astype(np.float32))
                occluded_ok=True
                for i,low in enumerate((0,1.5,4)):
                    blocked=(image[1]>0)&(d<low)
                    occluded_ok &= bool(np.all(bands[0,i,2].numpy()[blocked]==0))
                # Quantized first returns must survive without changing their masks.
                hit_identity=np.array_equal(image[1].astype(bool),scan.hit_mask)
                point_error=float(np.linalg.norm((d[scan.hit_mask]-scan.range_m[scan.hit_mask])[:,None]*scan.directions_world[scan.hit_mask],axis=-1).max()) if scan.hit_mask.any() else 0.
                geometry.append(dict(map_id=spec.map_id,split=spec.split,position=pos,hit_rays=int(scan.hit_mask.sum()),near_hit_rays=int(near.sum()),
                                     nearest_preserved=bool(nearest_ok),surface_membership_once=bool(hit_once),occluded_not_free=bool(occluded_ok),
                                     hit_mask_identical=bool(hit_identity),quantized_point_error_m=point_error))
                # Probes target only identifiable first-return geometry, not hidden global truth.
                az=np.arctan2(directions[...,1],directions[...,0]);el=np.arcsin(directions[...,2])
                front=(abs(az)<np.deg2rad(25))&(abs(el)<np.deg2rad(20))
                free_front=front&(d>3.)
                fraction=float(free_front.sum()/front.sum())
                center_az=float(az[free_front].mean()/np.deg2rad(25)) if free_front.any() else 0.
                center_el=float(el[free_front].mean()/np.deg2rad(20)) if free_front.any() else 0.
                detected_pole=any(np.any(scan.hit_body==body) for body,o in zip(bodies,spec.obstacles) if o.name.startswith('rod_'))
                labels=[float(d[image[1]>0].min()/8) if scan.hit_mask.any() else 1., float(detected_pole), fraction,center_az,center_el]
                records.append(dict(map_id=spec.map_id,split=spec.split,scenario=spec.scenario,position=pos,
                                    geometry_sha256=data['geometry_sha256'],labels=labels))
                images.append(image)
                if data['scenario'] not in examples:
                    examples[data['scenario']]=dict(image=image,bands=bands[0].numpy(),summary=summary,points=scan.pointcloud(),origin=scan.origin_world,
                                                   data=data,position=pos)
            print(json.dumps(dict(stage='geometry',map_id=spec.map_id,observations=len(images))),flush=True)
    finally:p.disconnect(client)
    array=np.stack(images)
    np.savez_compressed(out/'probe_dataset.npz',image=array,labels=np.array([r['labels'] for r in records],np.float32))
    save(out/'probe_records.json',records);save(out/'geometry_cases.json',geometry)
    save(out/'dataset_protocol.json',dict(catalog_fingerprint=catalog['fingerprint'],maps=sorted({r['map_id'] for r in records}),positions=positions,
        split='original Step3 train/validation only; sealed test excluded',actor_access='raw sensor only; map/body IDs used by QA labels only',
        label_names=['nearest_observed_surface_range_div8','detected_rod_presence','front_visible_prefix_beyond3m_fraction',
                     'front_visible_prefix_azimuth_centroid_div25deg','front_visible_prefix_elevation_centroid_div20deg'],
        label_limits='centroid describes sampled free-prefix window, not a certified physical passage or minimum vehicle clearance',collection_s=perf_counter()-t0))
    return torch.from_numpy(array),records,geometry,examples

def fit_probe(train_x,test_x,train_y,test_y):
    """Same linear head/ridge for all representations; map splits fixed."""
    mean=train_x.mean(0);std=train_x.std(0);std[std<1e-6]=1
    a=(train_x-mean)/std;b=(test_x-mean)/std
    a=np.column_stack((a,np.ones(len(a))));b=np.column_stack((b,np.ones(len(b))))
    ym=train_y.mean(0);ys=train_y.std(0);ys[ys<1e-6]=1
    lam=1.;regular=np.eye(a.shape[1])*lam;regular[-1,-1]=0
    weights=np.linalg.solve(a.T@a+regular,a.T@((train_y-ym)/ys))
    prediction=b@weights*ys+ym;error=prediction-test_y
    denominator=((test_y-test_y.mean(0))**2).sum(0)
    scores=1-(error**2).sum(0)/np.maximum(denominator,1e-12)
    classification=prediction[:,1]>=.5
    target=test_y[:,1]>=.5
    return dict(mae=abs(error).mean(0).tolist(),r2=scores.tolist(),pole_accuracy=float((classification==target).mean()),
                pole_recall=float(classification[target].mean()) if target.any() else None,
                pole_false_positive=float(classification[~target].mean()) if (~target).any() else None),prediction

def probes(out,images,records,pretrained):
    train=np.array([r['split']=='train' for r in records]);test=~train
    y=np.array([r['labels'] for r in records]);rows=[];features={};bypass=NearBypass(LidarConfig().directions())
    for name,spec in VARIANTS[:4]:
        for seed in (43,44,45):
            torch.manual_seed(seed);encoder=Encoder(spec).eval();parts=[]
            with torch.no_grad():
                for x in images.split(32):
                    z=encoder(x)
                    if spec.bypass:z=torch.cat((z,bypass(x)),1)
                    parts.append(z.numpy())
            x=np.concatenate(parts);result,pred=fit_probe(x[train],x[test],y[train],y[test])
            rows.append(dict(variant=name,encoder_status='random_frozen_no_RL',seed=seed,features=x.shape[1],**result))
            np.savez_compressed(out/f'probe_{name}_seed{seed}.npz',features=x,predictions_validation=pred)
            if seed==43:features[name]=x
    # A frozen intermediate rv1 is contextual only; never rank it as a fair trained rv2 comparison.
    payload=torch.load(pretrained,weights_only=False,map_location='cpu');enc=RangeEncoder();enc.load_state_dict({k.removeprefix('encoder.'):v for k,v in payload['agent']['actor'].items() if k.startswith('encoder.')})
    for name,use_bypass in [('rv1_frozen_intermediate',False),('rv1_frozen_intermediate_plus_bypass',True)]:
        with torch.no_grad():
            values=[torch.cat((enc(x),bypass(x)),1).numpy() if use_bypass else enc(x).numpy() for x in images.split(32)]
        x=np.concatenate(values);result,pred=fit_probe(x[train],x[test],y[train],y[test])
        rows.append(dict(variant=name,encoder_status='frozen existing rv1 intermediate; no rv2 trained counterpart',seed=43,features=x.shape[1],**result))
        np.savez_compressed(out/f'probe_{name}.npz',features=x,predictions_validation=pred)
    save(out/'probe_results.json',dict(results=rows,train_examples=int(train.sum()),validation_examples=int(test.sum()),
          protocol='linear ridge lambda1; standardized features/targets using train only; 3 frozen initialization seeds; no encoder fitting; no navigation inference',
          warning='bypass changes readable feature dimension 64->112; direct nearest feature supplies its target; not proof CNN learned or RL benefits'))
    save(out/'probe_checkpoint.json',dict(path=str(pretrained),sha256=sha(pretrained),scope='actor encoder read only; no optimizer or Replay loaded'))

def thin_pair(out):
    """A single pixel rod is measured by real ray cast before feature comparisons."""
    client=p.connect(p.DIRECT);cfg=LidarConfig();row,col=15,0;d=cfg.directions()[row,col]
    origin=np.array([0,0,1.02]);center=origin+.8*d
    try:
        sensor=Lidar(cfg)
        empty=sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=0)
        shape=p.createCollisionShape(p.GEOM_SPHERE,radius=.008,physicsClientId=client)
        body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=center,physicsClientId=client)
        hit=sensor.scan(client_id=client,position=[0,0,1],quaternion_xyzw=[0,0,0,1],timestamp_s=0,obstacle_ids=[body])
        assert hit.hit_mask[row,col] and hit.hit_mask.sum()==1
        def convert(scan):return canonical_image(dict(range_m=scan.range_m,hit_mask=scan.hit_mask,scanned_mask=scan.scanned_mask))
        x=torch.from_numpy(np.stack((convert(empty),convert(hit))))
        near=NearBypass(cfg.directions());v=near(x)
        rows=[]
        for name,spec in VARIANTS[:4]:
            torch.manual_seed(43);enc=Encoder(spec)
            with torch.no_grad():z,detail=enc(x,True)
            rows.append(dict(variant=name,latent_l2_delta=float(torch.linalg.vector_norm(z[1]-z[0])),
                             near_early_feature_l2_delta=float(torch.linalg.vector_norm(detail['early'][0][1]-detail['early'][0][0])) if detail['early'] else None))
        save(out/'single_hit_pair.json',dict(sensor='one real hit, 8mm radius sphere simulating a one-pixel small obstacle; not a continuous rod',
             ray=[row,col],hit_rays=int(hit.hit_mask.sum()),distance_m=float(hit.range_m[row,col]),
             changed_bypass_distance_sectors=int((v[0,::2]!=v[1,::2]).sum()),encoder_deltas=rows,
             caveat='nonzero delta shows sensitivity only; cannot prove decodability or safety'))
        np.savez_compressed(out/'single_hit_pair.npz',image=x.numpy(),bypass=v.numpy())
    finally:p.disconnect(client)

def benchmark(out,images,parent,own_sha):
    rows=[];cfg=copy.deepcopy(load_config()['td3']);bypass=NearBypass(LidarConfig().directions())
    batch=images[:1].repeat(256,1,1,1);obs=dict(state=torch.zeros(1,18),image=images[:1])
    with torch.no_grad():
        preprocess=dict(radial_b1=timing(lambda:radial_images(images[:1])),radial_b256=timing(lambda:radial_images(batch),5),
                        bypass_b1=timing(lambda:bypass(images[:1])),bypass_b256=timing(lambda:bypass(batch),5))
    del batch
    for name,spec in VARIANTS:
        torch.manual_seed(43);c=make_contract(spec,parent,own_sha);agent=RangeTD3(cfg,c,spec)
        replay=SuccessReplay(c,capacity=8,success_capacity=4)
        sample=dict(state=np.zeros(18,np.float32),range_m=images[0,0].numpy()*8,
                    hit_mask=images[0,1].numpy().astype(bool),scanned_mask=images[0,2].numpy().astype(bool))
        sample['range_m'][~sample['scanned_mask']]=np.nan
        for i in range(6):replay.add(sample,np.zeros(3,np.float32),sample,.1,False,False)
        replay.finish(True,10)
        with torch.no_grad():inference=timing(lambda:agent.actor(obs),20)
        a_macs=macs(agent.actor,obs);enc_macs=macs(agent.actor.features.encoder,obs['image'])
        before={k:v.detach().clone() for k,v in agent.actor.features.encoder.named_parameters()}
        logs=[]
        for i in range(4):
            t=perf_counter();diag=agent.update(replay)
            logs.append(dict(update=i+1,wall_ms=(perf_counter()-t)*1000,**diag))
        change=float(torch.sqrt(sum((v.detach()-before[k]).square().sum() for k,v in agent.actor.features.encoder.named_parameters())))
        row=dict(variant=name,encoder_parameters=count(agent.actor.features.encoder),actor_parameters=count(agent.actor),critic_parameters=count(agent.critic),
                 actor_input_dim=agent.actor.features.input_dim,encoder_macs=enc_macs,encoder_conv_linear_flops_2mac=2*enc_macs,
                 actor_macs=a_macs,inference_b1=inference,updates=logs,batch_size=cfg['batch_size'],actor_encoder_parameter_delta_l2=change,
                 process_cumulative_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        rows.append(row);save(out/'cost_partial.json',rows)
        # Engineering checkpoint + raw Replay, unique variant names. Not a training run.
        torch.save(agent.checkpoint(),out/f'{name}_engineering_checkpoint.pt',pickle_protocol=4);replay.save(out/f'{name}_engineering_replay.pt')
        save(out/f'{name}_contract.json',c)
        print(json.dumps(dict(stage='cost',variant=name,encoder_parameters=row['encoder_parameters'],max_update_ms=max(x['wall_ms'] for x in logs))),flush=True)
        del agent,replay;gc.collect()
    raw_bytes=2*(4096*2+2*512)+2*18*4+3*4+4+2+8+2*8
    save(out/'cost.json',dict(device=platform.processor(),platform=platform.platform(),torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),
         torch_threads=torch.get_num_threads(),co_running_rv1=True,timing_context='single CPU thread, rv1 continues on same CPU; descriptive wall-clock, not isolated hardware ranking',
         flops_definition='2 * MACs of Conv2d and Linear; excludes padding, activation, feature construction/scatter, backward and optimizer',
         preprocessing=preprocess,variants=rows,raw_transition_bytes=raw_bytes,main_60000_bytes=raw_bytes*60000,success_20000_bytes=raw_bytes*20000,
         compressed_b256_state_next_bytes=2*256*(8192+1024+72),decoded_images_b256_state_next_bytes=2*256*3*4096*4,
         radial_images_b256_one_side_bytes=256*3*5*4096*4,near_bypass_b256_bytes=256*48*4,
         memory_caveat='RSS is process cumulative peak, not per-model allocation or isolated peak; autograd intermediates and optimizer need extra memory'))

def plot(out,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for group,example in examples.items():
        fig,axes=plt.subplots(3,5,figsize=(15,7),constrained_layout=True)
        for i in range(3):
            for j in range(5):
                im=axes[i,j].imshow(example['bands'][i,j],origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
                axes[i,j].set_title(['hit in bin','local hit depth','visible free fraction','blocked before bin','scanned'][j] if i==0 else '')
                if j==0:axes[i,j].set_ylabel(['Near 0-1.5m','Mid 1.5-4m','Far 4-8m'][i])
                axes[i,j].set_xticks([0,32,64,96,127],['0','90','180','270','357'])
        fig.suptitle(f'Group {group}: first-return radial encoding (azimuth deg, elevation rows)')
        fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.012);fig.savefig(out/f'channels_{group}.png',dpi=140);plt.close(fig)
        fig=plt.figure(figsize=(14,4),constrained_layout=True)
        ax=fig.add_subplot(131);ax.imshow(example['image'][0]*8,origin='lower',aspect='auto',vmin=0,vmax=8);ax.set_title('Range image [m]')
        ax=fig.add_subplot(132,projection='3d');points=example['points'];ax.scatter(*points.T,s=3,c=points[:,2],cmap='viridis');ax.set_title('Measured first-return XYZ');ax.set_xlabel('world x [m]');ax.set_ylabel('world y [m]');ax.set_zlabel('z [m]')
        ax=fig.add_subplot(133);ax.imshow(example['summary'][:,0].reshape(3,8),origin='lower',vmin=0,vmax=1);ax.set_title('Bypass min range / 1.5m');ax.set_xlabel('yaw sector 0..7');ax.set_ylabel('lower / middle / upper')
        fig.savefig(out/f'geometry_{group}.png',dpi=140);plt.close(fig)
        np.savez_compressed(out/f'example_{group}.npz',image=example['image'],bands=example['bands'],bypass=example['summary'],points=example['points'])

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(43)
    own=source_manifest(Path(__file__).parent);save(out/'source_hashes.json',own)
    formal=ROOT/'experiments/rv1_forest_formal_v2/outputs/formal_seed43_v2_verified'
    parent=json.loads((formal/'formal_config.json').read_text())['sensor_contract']
    checkpoint=formal/'checkpoints/training_step040000.pt'
    if not checkpoint.exists():raise FileNotFoundError('predeclared frozen 40k rv1 probe checkpoint absent')
    catalog_path=ROOT/'experiments/rv1_range_cnn/outputs/step3_v3/challenge_catalog.json'
    catalog=json.loads(catalog_path.read_text());save(out/'input_catalog.json',dict(path=str(catalog_path),sha256=sha(catalog_path),fingerprint=catalog['fingerprint']))
    images,records,cases,examples=collect(out,catalog)
    thin_pair(out);probes(out,images,records,checkpoint);benchmark(out,images,parent,digest(own));plot(out,examples)
    valid=[r for r in cases if 'skipped' not in r]
    save(out/'geometry_gate.json',dict(cases=len(valid),skipped=len(cases)-len(valid),
         pass_all=all(all(r[k] for k in ('nearest_preserved','surface_membership_once','occluded_not_free','hit_mask_identical')) for r in valid),
         quantized_point_error_max_m=max(r['quantized_point_error_m'] for r in valid),
         navigation_benefit='not evaluated; no formal training',gpu_consistency='SKIP CUDA unavailable',angular_downsampling='DEFERRED by agreed scope'))
    print('AUDIT_COMPLETE',str(out),flush=True)

if __name__=='__main__':main(sys.argv[1])
