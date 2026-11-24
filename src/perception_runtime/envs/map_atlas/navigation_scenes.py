"""Four local navigation prototypes. No imported benchmark assets or RL changes."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .spec import MapSpec
from .navigation3d import find_route_3d

SCENARIOS = ('single_gap', 'vertical_slalom', 'curved_cave', 't_junction_mine')


@dataclass(frozen=True)
class NavigationSceneConfig:
    scenario: str
    length: float = 30.
    width: float = 10.
    height: float = 2.75
    wall_thickness: float = .20
    gap_width: float = 1.20
    gap_height: float = 1.30
    cave_width: float = 2.10
    cave_height: float = 1.60
    cave_segment_length: float = .30
    mine_width: float = 2.40
    sensor_range: float = 1.75
    episode_len_sec: float = 180.

    def validate(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f'Unknown scenario: {self.scenario}')
        for name, value in asdict(self).items():
            if name == 'scenario' or value is None:
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.length < 10 or self.height > 2.75 or self.height < 2.6:
            raise ValueError('Prototype requires length >=10 and height in [2.6,2.75] for existing Full bounds')
        width = self.width
        if width < 6 or self.wall_thickness > .3:
            raise ValueError('Prototype requires width >=6 and wall_thickness <=.3')
        if self.gap_width < .9 or self.gap_width > width - 3 or not .9 <= self.gap_height <= self.height - .6:
            raise ValueError('Opening dimensions cannot satisfy prototype clearance')
        if not 1.6 <= self.cave_width <= width - 2 or not 1.3 <= self.cave_height <= self.height - .8:
            raise ValueError('Invalid cave cross section')
        if not .15 <= self.cave_segment_length <= .6:
            raise ValueError('cave_segment_length must be in [.15,.6]')
        if not 1.8 <= self.mine_width <= min(width, self.length) / 2:
            raise ValueError('Invalid mine corridor width')


def generate_navigation_scene(config, *, seed=43):
    config.validate()
    rng = np.random.default_rng(seed)
    length, width, height = config.length, config.width, config.height
    hx, hy, t = length / 2, width / 2, config.wall_thickness
    floor, ceiling = .15, height - .20
    obstacles = []
    def box(name, lo, hi, color=(.43,.47,.52,1)):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if np.any(hi <= lo):
            raise ValueError(f'Degenerate box: {name}')
        obstacles.append(dict(kind='box', name=name, position=((lo+hi)/2).tolist(),
                              half_extents=((hi-lo)/2).tolist(), color=list(color)))
    def caps():
        box('floor', [-hx,-hy,0], [hx,hy,floor], (.38,.34,.28,1))
        box('ceiling', [-hx,-hy,ceiling], [hx,hy,height])
    def perimeter():
        box('boundary_west', [-hx,-hy,floor], [-hx+t,hy,ceiling])
        box('boundary_east', [hx-t,-hy,floor], [hx,hy,ceiling])
        box('boundary_south', [-hx,-hy,floor], [hx,-hy+t,ceiling])
        box('boundary_north', [-hx,hy-t,floor], [hx,hy,ceiling])
    start, goal = [-hx+1,0,1.25], [hx-1,0,1.25]
    details = {}
    caps()
    if config.scenario == 'single_gap':
        perimeter()
        offset = float(rng.uniform(.55, 1.0) * rng.choice([-1,1]))
        left, right = offset-config.gap_width/2, offset+config.gap_width/2
        bottom, top = 1.25-config.gap_height/2, 1.25+config.gap_height/2
        box('gap_left', [-t/2,-hy,floor], [t/2,left,ceiling], (.68,.48,.25,1))
        box('gap_right', [-t/2,right,floor], [t/2,hy,ceiling], (.68,.48,.25,1))
        box('gap_sill', [-t/2,left,floor], [t/2,right,bottom], (.68,.48,.25,1))
        box('gap_lintel', [-t/2,left,top], [t/2,right,ceiling], (.68,.48,.25,1))
        details['opening'] = dict(x=0, y=[left,right], z=[bottom,top])
    elif config.scenario == 'vertical_slalom':
        perimeter()
        gates = []
        for i, x in enumerate(np.array([-.22,0,.22])*length):
            x += float(rng.uniform(-.10,.10))
            if i % 2 == 0:
                cut = float(rng.uniform(1.40,1.48))
                box(f'gate_{i}_climb', [x-t/2,-hy,floor], [x+t/2,hy,cut], (.73,.41,.22,1))
                limits = [cut,ceiling]
            else:
                cut = float(rng.uniform(1.03,1.10))
                box(f'gate_{i}_descend', [x-t/2,-hy,cut], [x+t/2,hy,ceiling], (.28,.49,.67,1))
                limits = [floor,cut]
            gates.append(dict(x=x, opening_z=limits))
        details['gates'] = gates
        details['constant_altitude_impossible'] = max(g['opening_z'][0] for g in gates) > min(g['opening_z'][1] for g in gates)
    elif config.scenario == 'curved_cave':
        # A solid stepped shell: adjoining slabs have no gaps. Not a Perlin port.
        count = int(np.ceil(length/config.cave_segment_length))
        edges = np.linspace(-hx,hx,count+1)
        centers = (edges[:-1]+edges[1:])/2
        phase = float(rng.uniform(-.35,.35))
        cy = 1.45*np.sin(2*np.pi*centers/length + phase)
        cz = 1.35+.25*np.sin(2*np.pi*centers/length-.7+phase)
        widths = config.cave_width + rng.uniform(-.10,.10,count)
        heights = config.cave_height + rng.uniform(-.08,.08,count)
        profile = []
        for i, x in enumerate(centers):
            left,right = cy[i]-widths[i]/2, cy[i]+widths[i]/2
            bottom,top = max(floor,cz[i]-heights[i]/2), min(ceiling,cz[i]+heights[i]/2)
            x0,x1 = edges[i],edges[i+1]
            color = (.40+.04*(i%3),.36+.03*(i%3),.31+.02*(i%3),1)
            box(f'cave_{i}_left',[x0,-hy,floor],[x1,left,ceiling],color)
            box(f'cave_{i}_right',[x0,right,floor],[x1,hy,ceiling],color)
            if bottom > floor:
                box(f'cave_{i}_floor',[x0,left,floor],[x1,right,bottom],color)
            if top < ceiling:
                box(f'cave_{i}_roof',[x0,left,top],[x1,right,ceiling],color)
            profile.append(dict(x=float(x),y=float(cy[i]),z=float(cz[i]),width=float(widths[i]),floor=float(bottom),ceiling=float(top)))
        box('cave_end_west',[-hx,-hy,floor],[-hx+t,hy,ceiling])
        box('cave_end_east',[hx-t,-hy,floor],[hx,hy,ceiling])
        for point in (start,goal):
            i = int(np.argmin(abs(centers-point[0])))
            point[1],point[2] = float(cy[i]),float(cz[i])
        details.update(profile=profile, representation='solid axis-aligned segmented rock shell; no mesh assets')
    else:
        # T-shaped free space = vertical stem union horizontal crossbar.
        half = config.mine_width/2
        junction_y = hy * .4
        bar_low,bar_high = junction_y-half,junction_y+half
        stem_bottom,bar_end = -hy+1,hx-1
        box('mine_top',[-hx,bar_high,floor],[hx,hy,ceiling])
        box('mine_bottom',[-hx,-hy,floor],[hx,stem_bottom,ceiling])
        box('mine_left_mass',[-hx,stem_bottom,floor],[-half,bar_low,ceiling])
        box('mine_right_mass',[half,stem_bottom,floor],[hx,bar_low,ceiling])
        box('mine_left_end',[-hx,bar_low,floor],[-bar_end,bar_high,ceiling])
        box('mine_right_end',[bar_end,bar_low,floor],[hx,bar_high,ceiling])
        # Reproducible support pillars, kept outside the central flying lane.
        for i,y in enumerate(np.linspace(stem_bottom+.8,bar_low-.6,3)):
            inset = float(rng.uniform(.10,.18))
            for side in (-1,1):
                x = side*(half-inset/2)
                box(f'support_{i}_{side}',[x-inset/2,y-.10,floor],[x+inset/2,y+.10,ceiling],(.32,.35,.38,1))
        start = [0,stem_bottom+.6,1.25]
        goal = [bar_end-.6,junction_y,1.25]
        details.update(topology='one T junction, two terminal arms, no loops',
                       junction=[0,junction_y,1.25],unused_arm=[-bar_end+.6,junction_y,1.25])
    record = dict(map_id=f'{config.scenario}_{length:g}x{width:g}_seed{seed}_v1', source='local_procedural_navigation_v1',
                  scenario=config.scenario, split='development', start_pos=start,target_pos=goal,
                  xy_half_extent=max(hx,hy),xy_half_extents=[hx,hy],navigation_z_bounds=[0,height],
                  obstacle_sensor_range=config.sensor_range,episode_len_sec=config.episode_len_sec,
                  box_collision_mode='triangles' if config.scenario=='curved_cave' else 'primitive',
                  obstacles=obstacles,notes='Local geometry prototype; QA paths are privileged and are not actor inputs.')
    spec = MapSpec.from_dict(record,root=Path('.'))
    report,route = find_route_3d(spec)
    if not report.reachable:
        raise ValueError(f'Generated map failed QA: {report.message}')
    geometry = json.dumps(obstacles,sort_keys=True,separators=(',',':')).encode()
    meta = dict(scene_sha256=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),generator_version=1,seed=int(seed),config=asdict(config),geometry_sha256=hashlib.sha256(geometry).hexdigest(),
                details=details,qa=dict(message=report.message,resolution=.2,vehicle_radius=.06,margin=.10,
                                      path_length=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())),
                bounds_m=[length,width,height],asset_origin='original local procedural geometry')
    return record,meta,route

