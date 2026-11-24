"""Seeded challenge geometry, independent of policy outcomes. QA never feeds Actor."""
from dataclasses import replace,asdict
import hashlib,json
from pathlib import Path
import numpy as np
from envs.map_atlas.spec import MapSpec,StaticObstacleSpec
from envs.map_atlas.navigation3d import find_route_3d,signed_clearance

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class ChallengeMapSpec(MapSpec):
    def validate_route(self,*,drone_radius=.07,resolution=.10):
        # Only map admission uses 3D truth. Keep navigation_z_bounds=None on the
        # running map to retain dv0's existing proximity reward implementation.
        clone=replace(self,navigation_z_bounds=(.08,2.75))
        return find_route_3d(clone,vehicle_radius=drone_radius,margin=.03,resolution=resolution)[0]

def box(name,position,half):return dict(kind='box',name=name,position=list(position),half_extents=list(half),color=[.40,.47,.55,1.])
def pole(name,position,radius,height):return dict(kind='cylinder',name=name,position=list(position),radius=radius,height=height,color=[.3,.5,.3,1.])

VARIANTS={'A':['thin_04','thin_08','thin_16'],
          'B':['opening_low','opening_high','opening_thick'],
          'C':['over','under','alternating'],
          'D':['mixed','occluded'],
          'E':['unequal_sides','over_or_side','fork_deadend']}
SPLIT_SEEDS={'train':[310031,310032],'validation':[310131,310132],'test':[310231,310232]}

def generate(group,variant,split,seed):
    rng=np.random.default_rng(seed);j=float(rng.uniform(-.10,.10));obstacles=[]
    if group=='A':
        diameter={'thin_04':.04,'thin_08':.08,'thin_16':.16}[variant]
        obstacles=[pole(f'rod_{i}',[x,y+j,1.4],diameter/2,2.8) for i,(x,y) in enumerate([(-1.8,.05),(0.,-.22),(1.7,.28)])]
    elif group=='B':
        width,height,center,thickness={'opening_low':(.65,.75,1.,.20),'opening_high':(.85,.80,1.85,.20),'opening_thick':(.75,.85,1.3,.8)}[variant]
        y=j;bottom=center-height/2;top=center+height/2
        obstacles=[box('left_jamb',[0,(-3+y-width/2)/2,1.4],[thickness/2,(3+y-width/2)/2,1.4]),
                   box('right_jamb',[0,(3+y+width/2)/2,1.4],[thickness/2,(3-y-width/2)/2,1.4]),
                   box('sill',[0,y,bottom/2],[thickness/2,width/2,bottom/2]),
                   box('lintel',[0,y,(top+2.8)/2],[thickness/2,width/2,(2.8-top)/2])]
    elif group=='C':
        if variant in ('over','alternating'):
            obstacles.append(box('lower_barrier',[-.7 if variant=='alternating' else 0,0,.64+j/2],[.17,3,.64+j/2]))
        if variant in ('under','alternating'):
            bottom=.78+j
            obstacles.append(box('upper_barrier',[.9 if variant=='alternating' else 0,0,(2.8+bottom)/2],[.22,3,(2.8-bottom)/2]))
    elif group=='D':
        # Positions and dimensions come from a fixed geometric design, with
        # seeded jitter. No rejection/filtering based on dv0 rollouts.
        for i,x in enumerate([-2.,-.8,.5,1.8,2.7]):
            for k,y in enumerate([-1.65,-.55,.65,1.7]):
                r=float(rng.uniform(.07,.21));xx=x+float(rng.uniform(-.12,.12));yy=y+float(rng.uniform(-.12,.12))
                h=float(rng.uniform(1.35,2.8));obstacles.append(pole(f'tree_{i}_{k}',[xx,yy,h/2],r,h))
        if variant=='occluded':
            obstacles.extend([box('occluder',[-1.1,-.45,1.25],[.13,.4,1.25]),box('offset_beam',[1.2,.5,1.8],[.2,.7,.12])])
    elif variant=='unequal_sides':obstacles=[box('asymmetric_wall',[0,.42+j,1.4],[.22,1.15,1.4])]
    elif variant=='over_or_side':obstacles=[box('low_wide_wall',[0,.3+j,.66],[.24,1.35,.66])]
    else:
        obstacles=[box('branch_upper',[-.1,1.05+j,1.4],[1.25,.12,1.4]),
                   box('branch_lower',[-.1,-1.05+j,1.4],[1.25,.12,1.4]),
                   box('dead_end',[1.03,-.5+j,1.4],[.12,.46,1.4]),
                   box('divider',[-.2,0+j,1.4],[1.,.12,1.4])]
    return dict(map_id=f'rv1_{group}_{variant}_{split}_{seed}',source='rv1_seeded_3d_primitives_v1',
                scenario=group,split=split,start_pos=[-4,0,1],target_pos=[4,0,1],xy_half_extent=5.,
                xy_half_extents=[5.,3.],obstacle_sensor_range=8.,episode_len_sec=40.,
                box_collision_mode='primitive',obstacles=obstacles,
                notes='Fixed before policy evaluation; original start, no course relocation.',
                benchmark=dict(group=group,variant=variant,seed=seed))

def spec_from(data):return ChallengeMapSpec.from_dict(data,root=Path('.'))

def geometric_fingerprint(data):
    return digest({k:data[k] for k in ('start_pos','target_pos','xy_half_extents','obstacles','box_collision_mode')})

def qa_route(spec,**kwargs):
    report,path=find_route_3d(replace(spec,navigation_z_bounds=(.08,2.75)),vehicle_radius=.07,margin=.03,resolution=.1,**kwargs)
    return dict(reachable=report.reachable,message=report.message,path=path.tolist(),
                length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()) if len(path) else None,
                minimum_surface_clearance_m=float(signed_clearance(spec,path).min()) if len(path) else None,
                policy_access=False,vehicle_radius_m=.07,margin_m=.03,resolution_m=.1)

def alternative_routes(spec):
    """Constrained offline routes through lateral / vertical waypoint regions."""
    output=[]
    for label,waypoint in [('left',[0,2.15,1.]),('right',[0,-2.15,1.]),('over',[0,0,2.25])]:
        a=qa_route(replace(spec,target_pos=tuple(waypoint)))
        b=qa_route(replace(spec,start_pos=tuple(waypoint)))
        valid=a['reachable'] and b['reachable']
        output.append(dict(label=label,feasible=bool(valid),length_m=a['length_m']+b['length_m'] if valid else None,
                           path=a['path']+b['path'][1:] if valid else [],note='Waypoint-conditioned route, not exhaustive homotopy enumeration'))
    return output

def build_catalog(destination):
    maps=[]
    for split,seeds in SPLIT_SEEDS.items():
        for group,variants in VARIANTS.items():
            for variant in variants:
                for seed in seeds:
                    data=generate(group,variant,split,seed);data['geometry_sha256']=geometric_fingerprint(data)
                    maps.append(data)
    hashes=[x['geometry_sha256'] for x in maps]
    if len(set(hashes))!=len(hashes):raise RuntimeError('duplicate challenge geometry across splits')
    catalog=dict(schema='rv1_challenge_v1',split_seeds=SPLIT_SEEDS,variants=VARIANTS,maps=maps,
                 selection_rule='all predefined variants x seeds; never filtered by policy outcome',
                 test_policy='geometry QA only in Step3, no test policy rollout/checkpoint selection')
    catalog['fingerprint']=digest(catalog);Path(destination).write_text(json.dumps(catalog,indent=2))
    return catalog
