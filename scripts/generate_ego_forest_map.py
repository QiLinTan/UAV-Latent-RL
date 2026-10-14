"""Generate, render, and physically verify one EGO-inspired forest instance."""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

from envs.map_atlas.ego_forest import EgoForestConfig, generate_ego_forest
from envs.map_atlas import MapSpec
from envs.learned_upper_dslpid import LearnedUpperDSLPIDForestEnv
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def plot_map(record, route, folder, *, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(12, 6))
    half_x, half_y = record['xy_half_extents']
    ax.add_patch(Rectangle((-half_x,-half_y),2*half_x,2*half_y,fill=False,lw=2,color='#344054'))
    for obstacle in record['obstacles']:
        ax.add_patch(Circle(obstacle['position'][:2],obstacle['radius'],fc='#846344',ec='#513c26',alpha=.9))
    start, goal = np.asarray(record['start_pos']), np.asarray(record['target_pos'])
    ax.plot([start[0],goal[0]],[start[1],goal[1]],color='#dd9b36',ls=':',label='Straight start-to-goal reference')
    ax.plot(route[:,0],route[:,1],color='#64748b',lw=1.5,ls='--',label='Geometry QA route (not a policy rollout)')
    ax.scatter(*start[:2],s=110,c='#18976b',zorder=5,label=f'Start {tuple(start)}')
    ax.scatter(*goal[:2],s=110,c='#d34d4d',marker='*',zorder=5,label=f'Goal {tuple(goal)}')
    ax.set(xlim=(-half_x-.5,half_x+.5),ylim=(-half_y-.5,half_y+.5),xlabel='X (m)',ylabel='Y (m)',
           title=f"EGO-inspired random forest | {2*half_x:g} x {2*half_y:g} m | {len(record['obstacles'])} trees | seed {seed}")
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=9)
    fig.tight_layout();fig.savefig(folder/'map_xy.png',dpi=180,bbox_inches='tight');plt.close(fig)


def verify_and_render(record, folder, *, seed=43, gui=False):
    spec=MapSpec.from_dict(record,root=folder)
    env=LearnedUpperDSLPIDForestEnv(map_spec=spec,obs=ObservationType.KIN,act=ActionType.RPM,
        pyb_freq=240,ctrl_freq=120,gui=gui,range_azimuth_count=32,range_elevation_degrees=(-30,0,30),
        semantic_history_length=8,max_reference_speed=.8,max_reference_acceleration=1.,
        max_reference_vertical_speed=.3,max_reference_vertical_acceleration=.5)
    try:
        observation,info=env.reset(seed=seed)
        assert observation.shape==(1,917) and np.isfinite(observation).all()
        assert info['num_obstacles']==len(spec.obstacles)
        assert env.EPISODE_LEN_SEC==record['episode_len_sec']
        assert not env.base_env._checkCollision()
        # Test an actual world ray and physical contact against the same cylinder.
        first=spec.obstacles[0]; body=env.base_env._tree_ids[0]
        center=np.array([first.position[0],first.position[1],1.])
        ray_start=center-np.array([first.radius+.3,0,0])
        hit=p.rayTest(ray_start.tolist(),center.tolist(),physicsClientId=env.CLIENT)[0]
        assert hit[0]==body
        drone=int(env.base_env.DRONE_IDS[0])
        p.resetBasePositionAndOrientation(drone,center.tolist(),[0,0,0,1],physicsClientId=env.CLIENT)
        p.performCollisionDetection(physicsClientId=env.CLIENT)
        env.base_env._updateAndStoreKinematicInformation()
        contact=env.base_env._checkCollision()
        assert contact
        env.reset(seed=seed)
        lower_ticks=0
        for _ in range(60):
            obs,reward,terminated,truncated,step=env.step(np.zeros((1,3),dtype=np.float32))
            assert obs.shape==(1,917) and np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated and not truncated and step['reference_packet_valid']
            assert step['lower_steps_executed']==8
            lower_ticks+=step['lower_steps_executed']
        assert lower_ticks==480
        # Verify the short axis is enforced, independent of the legacy scalar extent.
        state=env.base_env._getDroneStateVector(0).copy()
        state[:3]=[0,spec.bounds_xy[1]+.01,1]
        assert env.base_env._classifyDoneReason(state,collision=False)=='xy_bound'
        state[:3]=[spec.bounds_xy[0]+.01,0,1]
        assert env.base_env._classifyDoneReason(state,collision=False)=='xy_bound'
        state[:3]=[0,0,1]
        assert env.base_env._classifyDoneReason(state,collision=False)=='running'
        env.reset(seed=seed)
        # Visual annotations have no collision bodies and never enter sensing.
        half_x, half_y = spec.bounds_xy
        for position, extents in (
            ([0,-half_y,.025],[half_x,.025,.025]),
            ([0,half_y,.025],[half_x,.025,.025]),
            ([-half_x,0,.025],[.025,half_y,.025]),
            ([half_x,0,.025],[.025,half_y,.025]),
        ):
            visual=p.createVisualShape(p.GEOM_BOX,halfExtents=extents,rgbaColor=[.15,.2,.3,1],physicsClientId=env.CLIENT)
            p.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,basePosition=position,physicsClientId=env.CLIENT)
        for position,color in ((spec.start_pos,[.1,.8,.3,1]),(spec.target_pos,[.9,.15,.15,1])):
            visual=p.createVisualShape(p.GEOM_SPHERE,radius=.18,rgbaColor=color,physicsClientId=env.CLIENT)
            p.createMultiBody(baseMass=0,baseVisualShapeIndex=visual,basePosition=position,physicsClientId=env.CLIENT)
        view=p.computeViewMatrix([17,-18,23],[0,0,1],[0,0,1])
        projection=p.computeProjectionMatrixFOV(fov=52,aspect=16/9,nearVal=.1,farVal=100)
        camera=p.getCameraImage(1600,900,viewMatrix=view,projectionMatrix=projection,
                               renderer=p.ER_TINY_RENDERER,physicsClientId=env.CLIENT)
        Image.fromarray(np.asarray(camera[2],dtype=np.uint8).reshape(900,1600,4)[:,:,:3]).save(folder/'map_pybullet.png')
        if gui:
            p.resetDebugVisualizerCamera(23,35,-55,[0,0,1],physicsClientId=env.CLIENT)
            input('Scene verified. Press Enter to close the viewer. ')
        return dict(observation_dim=917,upper_hz=env.CTRL_FREQ,lower_hz=env.lower_ctrl_freq,
                    hover_upper_steps=60,lower_ticks=lower_ticks,episode_len_sec=env.EPISODE_LEN_SEC,
                    reset_collision=False,ray_hit_same_collision_tree=True,contact_inside_tree=True,
                    rectangular_x_y_bounds_checked=True,reference_packet_valid=True,finite_observations_rewards=True)
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--x_size',type=float,default=20.)
    parser.add_argument('--y_size',type=float,default=10.)
    parser.add_argument('--trees',type=int,default=40)
    parser.add_argument('--output_dir',type=Path)
    parser.add_argument('--register_catalog',type=Path)
    parser.add_argument('--gui',action='store_true')
    args=parser.parse_args()
    folder=args.output_dir or Path(f'data/map_atlas/generated/ego_forest_{args.x_size:g}x{args.y_size:g}_seed{args.seed}_v1')
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f'Refusing to overwrite a generated map: {folder}')
    config=EgoForestConfig(size_x=args.x_size,size_y=args.y_size,tree_count=args.trees)
    record,metadata,route=generate_ego_forest(config,seed=args.seed)
    folder.mkdir(parents=True,exist_ok=True)
    metadata['physical_validation']=verify_and_render(record,folder,seed=args.seed,gui=args.gui)
    plot_map(record,route,folder,seed=args.seed)
    (folder/'map_catalog.json').write_text(json.dumps({'schema_version':'uav_map_atlas_v1','maps':[record]},indent=2)+'\n')
    (folder/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (folder/'qa_route.json').write_text(json.dumps({'purpose':'privileged geometry QA only','positions':route.tolist()},indent=2)+'\n')
    if args.register_catalog:
        catalog=json.loads(args.register_catalog.read_text())
        if any(r['map_id']==record['map_id'] for r in catalog['maps']):
            raise FileExistsError(f"Map ID already registered: {record['map_id']}")
        catalog['maps'].append(record)
        args.register_catalog.write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps({'map_id':record['map_id'],'output_dir':str(folder),**metadata},indent=2))


if __name__=='__main__':
    main()
