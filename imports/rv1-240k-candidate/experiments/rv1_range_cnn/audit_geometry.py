"""Executable pre-CNN geometry/cost gate. Saves all measurements, including misses."""
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter
import json,csv,platform,sys
import numpy as np
import pybullet as p
from scipy.spatial import cKDTree
from envs.map_atlas.runtime import MapRuntime
from envs.map_atlas.spec import StaticObstacleSpec
from envs.map_atlas.navigation3d import signed_clearance
from .benchmark import build_catalog,spec_from,qa_route,alternative_routes,digest,box,pole
from .lidar import Lidar,LidarConfig,LAYOUTS
from .codec import RangeCodec
from .environment import RangeImageEnv
from .provenance import sources,contract

PROTOCOL=dict(version='v2_explicit_engine_tolerance_and_valid_pose',reconstruction_max_m=2e-6,quantization_extra_tolerance_m=1e-6,
              analytic_boundary_max_m=.02,geometry_colliders_required=True,
              engine_surface_uncertainty_m=.001,
              first_hit_depth='report strict ideal primitive error; gate first hit against +/-1mm surface envelope AND actual hit surface distance with unchanged 2cm bound',
              protocol_correction='v1 incorrectly sampled inside a wall and equated tangential first-hit depth jumps with surface error; v1 failures and source snapshot preserved',
              scan_p95_budget_ms=1000/15,environment_min_upper_steps_per_s=15.,
              thin_pole_detection='descriptive, no claim of guaranteed detection between rays',
              input='same 360deg x [-30,+30]deg and 8m; wide VFOV separate')

def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def expected_ranges(spec,origin,directions,cfg,ground=False):
    """Analytic first intersections with exact declared primitives; QA only."""
    d=directions.reshape(-1,3);origin=np.asarray(origin);best=np.full(len(d),np.inf)
    for obstacle in spec.obstacles:
        o=origin-np.asarray(obstacle.position)
        if obstacle.kind=='box':
            e=np.array(obstacle.half_extents)
            # Infinite slab intervals, including exactly parallel rays.
            with np.errstate(divide='ignore',invalid='ignore'):
                a=(-e-o)/d;b=(e-o)/d
            low=np.minimum(a,b);high=np.maximum(a,b)
            parallel=abs(d)<1e-12
            low=np.where(parallel,-np.inf,low);high=np.where(parallel,np.inf,high)
            outside=(parallel & (abs(o)>e)).any(axis=1)
            enter=low.max(axis=1);exit=high.min(axis=1)
            candidate=np.where((exit>=np.maximum(enter,cfg.origin_offset_m)) & ~outside,
                               np.where(enter>=cfg.origin_offset_m,enter,exit),np.inf)
        else:
            a=np.sum(d[:,:2]**2,axis=1);b=2*(d[:,:2]@o[:2]);cc=np.sum(o[:2]**2)-obstacle.radius**2
            discr=b*b-4*a*cc
            root=np.sqrt(np.maximum(0,discr));den=2*np.maximum(a,1e-30)
            candidate=np.full(len(d),np.inf)
            for t in ((-b-root)/den,(-b+root)/den):
                valid=(discr>=0)&(a>1e-12)&(t>=cfg.origin_offset_m)&(abs(o[2]+t*d[:,2])<=obstacle.height/2+1e-10)
                candidate=np.minimum(candidate,np.where(valid,t,np.inf))
            for z in (-obstacle.height/2,obstacle.height/2):
                with np.errstate(divide='ignore',invalid='ignore'):t=(z-o[2])/d[:,2]
                with np.errstate(invalid='ignore'):points=o[:2]+t[:,None]*d[:,:2]
                valid=(abs(d[:,2])>1e-12)&(t>=cfg.origin_offset_m)&(np.linalg.norm(points,axis=1)<=obstacle.radius+1e-10)
                candidate=np.minimum(candidate,np.where(valid,t,np.inf))
        best=np.minimum(best,candidate)
    if ground:
        with np.errstate(divide='ignore',invalid='ignore'):t=-origin[2]/d[:,2]
        best=np.minimum(best,np.where((d[:,2]<0)&(t>=cfg.origin_offset_m),t,np.inf))
    return np.where(best<=cfg.max_range_m,best,np.inf).reshape(cfg.shape)

def scan_world(sensor,client,position,obstacle_ids=(),ground_ids=(),yaw=0):
    return sensor.scan(client_id=client,position=position,quaternion_xyzw=p.getQuaternionFromEuler([0,0,yaw]),
                       timestamp_s=0,obstacle_ids=obstacle_ids,ground_ids=ground_ids)

def uncertainty_spec(spec,amount):
    obstacles=[]
    for o in spec.obstacles:
        if o.kind=='box':obstacles.append(replace(o,half_extents=tuple(np.array(o.half_extents)+amount)))
        else:obstacles.append(replace(o,radius=o.radius+amount,height=o.height+2*amount))
    return replace(spec,obstacles=tuple(obstacles))

def collision_qa(catalog,client):
    rows=[]
    for data in catalog['maps']:
        spec=spec_from(data)
        for obstacle in spec.obstacles:
            p.resetSimulation(physicsClientId=client)
            body=MapRuntime._create_body(obstacle,client_id=client)
            cd=p.getCollisionShapeData(body,-1,physicsClientId=client)[0];vd=p.getVisualShapeData(body,physicsClientId=client)[0]
            extent=np.array(obstacle.half_extents if obstacle.kind=='box' else [obstacle.radius,obstacle.radius,obstacle.height/2])
            center=np.array(obstacle.position);starts=[];ends=[]
            for axis in range(3):
                for sign in (-1,1):
                    v=np.zeros(3);v[axis]=sign
                    starts.append((center+v*(extent[axis]+.4)).tolist());ends.append(center.tolist())
            hits=p.rayTestBatch(starts,ends,physicsClientId=client)
            error=max(abs(np.array(h[3])[i//2]-center[i//2])-extent[i//2] for i,h in enumerate(hits))
            exact_axis=max(abs(abs(np.array(h[3])[i//2]-center[i//2])-extent[i//2]) for i,h in enumerate(hits))
            rows.append(dict(map_id=spec.map_id,obstacle=obstacle.name,kind=obstacle.kind,
                             visual_collision_type_equal=cd[2]==vd[2],visual_collision_dimensions_equal=bool(np.allclose(cd[3],vd[3])),
                             six_face_hits=all(h[0]==body for h in hits),axis_boundary_error_m=float(exact_axis)))
    return rows

def audit(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);save_json(out/'geometry_protocol.json',PROTOCOL)
    catalog=build_catalog(out/'challenge_catalog.json');save_json(out/'source_hashes_geometry.json',sources())
    qa=[]
    for data in catalog['maps']:
        spec=spec_from(data);row=dict(map_id=spec.map_id,split=spec.split,geometry_sha256=data['geometry_sha256'],**qa_route(spec))
        if spec.scenario=='E':row['alternatives']=alternative_routes(spec)
        qa.append(row)
    save_json(out/'offline_map_qa.json',qa)
    print(json.dumps(dict(stage='map_QA',maps=len(qa),reachable=sum(x['reachable'] for x in qa))),flush=True)
    client=p.connect(p.DIRECT)
    try:
        collision=collision_qa(catalog,client);write_csv(out/'collision_geometry.csv',collision)
        thin=[];vertical=[]
        # Common absolute phases, not separately chosen favourable phases per layout.
        for diameter in (.02,.04,.08,.16):
            for distance in (1.,2.,4.,6.):
                for phase in np.linspace(0,11.25,12,endpoint=False):
                    p.resetSimulation(physicsClientId=client);angle=np.deg2rad(phase)
                    obstacle=StaticObstacleSpec.from_dict(pole('thin',[distance*np.cos(angle),distance*np.sin(angle),1.4],diameter/2,2.8),root=out)
                    body=MapRuntime._create_body(obstacle,client_id=client)
                    for name,cfg in LAYOUTS.items():
                        s=scan_world(Lidar(cfg),client,[0,0,1],obstacle_ids=[body])
                        thin.append(dict(layout=name,diameter_m=diameter,distance_m=distance,azimuth_phase_deg=float(phase),
                                         detected=bool(s.hit_mask.any()),hit_rays=int(s.hit_mask.sum())))
        write_csv(out/'thin_pole_all_phases.csv',thin)
        for center_z in (.5,.8,1.1,1.4,1.7,2.):
            for thickness in (.04,.08,.16):
                p.resetSimulation(physicsClientId=client)
                obstacle=StaticObstacleSpec.from_dict(box('thin_beam',[3,0,center_z],[.08,.8,thickness/2]),root=out)
                body=MapRuntime._create_body(obstacle,client_id=client)
                for name,cfg in LAYOUTS.items():
                    s=scan_world(Lidar(cfg),client,[0,0,1],obstacle_ids=[body])
                    vertical.append(dict(layout=name,beam_center_z_m=center_z,thickness_m=thickness,detected=bool(s.hit_mask.any()),hit_rays=int(s.hit_mask.sum())))
        write_csv(out/'thin_beam_all_phases.csv',vertical)
        geometry=[];timings=[];images=out/'scans';images.mkdir(exist_ok=True)
        configs={**LAYOUTS,'4096_32x128_wide120':replace(LidarConfig(),vertical_min_deg=-60,vertical_max_deg=60)}
        for data in catalog['maps']:
            if data['split']!='validation':continue
            spec=spec_from(data);p.resetSimulation(physicsClientId=client)
            runtime=MapRuntime(spec);bodies=runtime.create_bodies(client_id=client)
            shape=p.createCollisionShape(p.GEOM_PLANE,physicsClientId=client);ground=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,physicsClientId=client)
            for pose_index,requested_position in enumerate(([-2.5,0,1.],[-1.1,.17,1.5])):
                position=np.array(requested_position)
                pose_adjusted=False
                if signed_clearance(spec,position[None])[0]<.12:
                    # QA pose correction is geometric and independent of policy.
                    route=np.array(next(r['path'] for r in qa if r['map_id']==spec.map_id))
                    position=route[np.argmin(np.linalg.norm(route-position,axis=1))];pose_adjusted=True
                for name,cfg in configs.items():
                    sensor=Lidar(cfg);start=perf_counter();s=scan_world(sensor,client,position,bodies,[ground]);duration=perf_counter()-start
                    expected=expected_ranges(spec,s.origin_world,s.directions_world,cfg,ground=True);gt=np.isfinite(expected)
                    both=gt&s.hit_mask
                    errors=abs(s.range_m[both]-expected[both]);reconstructed=s.pointcloud()
                    expanded=expected_ranges(uncertainty_spec(spec,.001),s.origin_world,s.directions_world,cfg,ground=True)
                    contracted=expected_ranges(uncertainty_spec(spec,-.001),s.origin_world,s.directions_world,cfg,ground=True)
                    lo=np.minimum(expanded,cfg.max_range_m);hi=np.minimum(contracted,cfg.max_range_m)
                    envelope_error=np.maximum(np.maximum(lo-s.range_m,s.range_m-hi),0)
                    surface_error=abs(signed_clearance(spec,reconstructed)) if len(reconstructed) else np.zeros(0)
                    # Ground is also a real first-hit surface, absent from MapSpec.
                    if len(reconstructed):surface_error=np.minimum(surface_error,abs(reconstructed[:,2]))
                    reconstruction=float(np.max(abs(reconstructed-s.raw_hit_world[s.hit_mask]))) if s.hit_mask.any() else 0
                    codec=RangeCodec(cfg.shape,cfg.max_range_m);t0=perf_counter();q,m=codec.encode(s.range_m,s.hit_mask,s.scanned_mask);decoded,h,sc=codec.decode(q,m);image=codec.policy_image(q,m);preprocess=perf_counter()-t0
                    quant=float(np.max(abs(s.range_m-decoded)))
                    row=dict(map_id=spec.map_id,group=spec.scenario,variant=data['benchmark']['variant'],pose_index=pose_index,layout=name,
                        rays=cfg.height*cfg.width,scan_ms=duration*1000,preprocess_ms=preprocess*1000,
                        hit_rays=int(s.hit_mask.sum()),ground_rays=int((s.hit_type==2).sum()),obstacle_rays=int((s.hit_type==3).sum()),
                        self_rays=int((s.hit_type==1).sum()),nohit_rays=int((~s.hit_mask & s.scanned_mask).sum()),
                        analytic_missed_rays=int((gt&~s.hit_mask).sum()),analytic_extra_rays=int((~gt&s.hit_mask).sum()),
                        boundary_error_max_m=float(errors.max()) if errors.size else 0.,boundary_error_mean_m=float(errors.mean()) if errors.size else 0.,
                        surface_error_max_m=float(surface_error.max()) if surface_error.size else 0.,
                        first_hit_envelope_error_max_m=float(envelope_error.max()),
                        qa_pose_adjusted=pose_adjusted,qa_pose_surface_clearance_m=float(signed_clearance(spec,s.origin_world[None])[0]),
                        reconstruction_error_max_m=reconstruction,quantization_error_max_m=quant,
                        hit_mask_unchanged=bool(np.array_equal(h,s.hit_mask)),scan_mask_unchanged=bool(np.array_equal(sc,s.scanned_mask)))
                    # Sample the opening on its wall plane, retaining explicit ray gaps.
                    if spec.scenario=='B':
                        sill=next(o for o in spec.obstacles if o.name=='sill');lintel=next(o for o in spec.obstacles if o.name=='lintel')
                        center_y=sill.position[1];half_width=sill.half_extents[1];bottom=sill.position[2]*2;top=lintel.position[2]-lintel.half_extents[2]
                        d=s.directions_world.reshape(-1,3)
                        forward=d[:,0]>1e-8;t=-s.origin_world[0]/np.where(forward,d[:,0],np.inf)
                        cross=s.origin_world+t[:,None]*d
                        opening=forward&(abs(cross[:,1]-center_y)<half_width)&(cross[:,2]>bottom)&(cross[:,2]<top)
                        pix=np.argwhere(opening.reshape(cfg.shape))
                        edge=np.vstack([np.c_[np.full(40,center_y-half_width),np.linspace(bottom,top,40)],np.c_[np.full(40,center_y+half_width),np.linspace(bottom,top,40)],
                                        np.c_[np.linspace(center_y-half_width,center_y+half_width,40),np.full(40,bottom)],np.c_[np.linspace(center_y-half_width,center_y+half_width,40),np.full(40,top)]])
                        candidates=cross[forward& (abs(cross[:,1])<=3)&(cross[:,2]>=0)&(cross[:,2]<=2.8),1:]
                        err=cKDTree(candidates).query(edge)[0] if len(candidates) else np.full(len(edge),10.)
                        row.update(opening_rays=int(opening.sum()),opening_rows=int(len(set(pix[:,0]))) if len(pix) else 0,
                                   opening_columns=int(len(set(pix[:,1]))) if len(pix) else 0,opening_boundary_sampling_p95_m=float(np.quantile(err,.95)))
                    geometry.append(row);timings.append(duration)
                    if pose_index==0:
                        np.savez_compressed(images/f'{spec.map_id}_{name}.npz',range_m=s.range_m,hit_mask=s.hit_mask,scanned_mask=s.scanned_mask,
                            directions_world=s.directions_world,origin_world=s.origin_world,xyz=reconstructed,
                            compressed_range=q,compressed_masks=m,hit_type=s.hit_type)
        write_csv(out/'scene_geometry.csv',[{k:r.get(k,'') for k in dict.fromkeys(k for rr in geometry for k in rr)} for r in geometry])
        perf=[]
        for name,cfg in LAYOUTS.items():
            env=RangeImageEnv(lidar_config=cfg)
            try:
                env.reset(seed=43);start=perf_counter()
                for _ in range(90):env.step([.15,0,0])
                elapsed=perf_counter()-start
                perf.append(dict(layout=name,upper_steps=90,wall_seconds=elapsed,steps_per_second=90/elapsed,
                    scans=env.lidar.scan_count,lower_ticks=env.controller.tick,physics_ticks=env.base_env.physics_ticks,
                    hidden_legacy_raycasts=env.base_env.ray_casts,self_hits=int((env.lidar.cache.hit_type==1).sum())))
            finally:env.close()
        write_csv(out/'environment_throughput.csv',perf)
        memory=[]
        for name,cfg in LAYOUTS.items():
            n=cfg.height*cfg.width;packed=2*((n+7)//8)
            memory.append(dict(layout=name,rays=n,float32_distance_pair_60k_GiB=2*60000*n*4/2**30,
                uint16_distance_pair_60k_GiB=2*60000*n*2/2**30,packed_mask_pair_60k_GiB=2*60000*packed/2**30,
                full_observation_pair_60k_GiB=2*60000*(n*2+packed+18*4)/2**30,
                unique_next_observation_chain_estimate_60k_GiB=60001*(n*2+packed+18*4)/2**30,
                quantization_bound_m=cfg.max_range_m/65534/2))
        write_csv(out/'replay_memory.csv',memory)
        aggregation=[]
        for name,cfg in configs.items():
            rows=[r for r in geometry if r['layout']==name];rods=[r for r in thin if r['layout']==name];beams=[r for r in vertical if r['layout']==name]
            aggregation.append(dict(layout=name,configuration=asdict(cfg),angular_resolution_deg=cfg.angular_resolution_deg,
                scan_ms_median=float(np.median([r['scan_ms'] for r in rows])),scan_ms_p95=float(np.quantile([r['scan_ms'] for r in rows],.95)),
                preprocess_ms_median=float(np.median([r['preprocess_ms'] for r in rows])),
                thin_pole_detection_rate=float(np.mean([r['detected'] for r in rods])) if rods else None,
                thin_beam_detection_rate=float(np.mean([r['detected'] for r in beams])) if beams else None,
                boundary_error_max_m=max(r['boundary_error_max_m'] for r in rows),
                surface_error_max_m=max(r['surface_error_max_m'] for r in rows),
                first_hit_envelope_error_max_m=max(r['first_hit_envelope_error_max_m'] for r in rows),
                analytic_missed_rays=sum(r['analytic_missed_rays'] for r in rows),analytic_extra_rays=sum(r['analytic_extra_rays'] for r in rows)))
        gates=dict(all_maps_feasible=all(r['reachable'] for r in qa),
            all_obstacles_real_colliders=all(r['six_face_hits'] and r['visual_collision_type_equal'] and r['visual_collision_dimensions_equal'] and r['axis_boundary_error_m']<.002 for r in collision),
            reconstruction=all(r['reconstruction_error_max_m']<=PROTOCOL['reconstruction_max_m'] for r in geometry),
            quantization_masks=all(r['hit_mask_unchanged'] and r['scan_mask_unchanged'] and r['quantization_error_max_m']<=8/65534/2+1e-6 for r in geometry),
            boundary_error=all(max(r['surface_error_max_m'],r['first_hit_envelope_error_max_m'])<=PROTOCOL['analytic_boundary_max_m'] for r in geometry),
            scan_budget=all(r['scan_ms_p95']<=PROTOCOL['scan_p95_budget_ms'] for r in aggregation),
            environment_budget=all(r['steps_per_second']>=15 and r['hidden_legacy_raycasts']==0 and r['self_hits']==0 and r['scans']==91 and r['physics_ticks']==1440 for r in perf))
        result=dict(status='PASS_GEOMETRY_COST' if all(gates.values()) else 'FAIL_GEOMETRY_COST',gates=gates,protocol=PROTOCOL,
            maps=len(catalog['maps']),catalog_fingerprint=catalog['fingerprint'],layouts=aggregation,memory=memory,environment=perf,
            source_sha256=digest(sources()),platform=platform.platform(),selection='4096_32x128 predeclared primary; no policy performance selection',
            contract=contract(LidarConfig(),catalog['fingerprint']),gpu='not measured here; report CPU and actual CUDA availability separately')
        save_json(out/'geometry_gate.json',result)
        print(json.dumps(result,indent=2),flush=True)
        return result
    finally:p.disconnect(client)

if __name__=='__main__':
    result=audit(sys.argv[1]);raise SystemExit(0 if result['status']=='PASS_GEOMETRY_COST' else 1)
