# v2-lite map atlas

`catalog_v1.json` is the experiment registry. It is intentionally small: the
first objective is to verify that a map can alter collision geometry, sensing,
start/goal states, and dataset split without changing the learned upper action
interface or the `ReferencePacket -> DSLPID` chain.

## Directory contract

```text
data/map_atlas/
├── catalog_v1.json          # committed map IDs, provenance, and splits
├── sources/                 # committed upstream revision records
├── raw/                     # downloaded upstream data; ignored by Git
├── converted/               # derived OBJ/STL collision meshes; ignored by Git
└── README.md
```

The shipped `*_proxy_v1` entries are local structural proxies, suitable for
adapter tests only. They are not DARPA SubT or FlightBench results and must not
be presented as such.

The initial SubT raw source is already present locally at
`raw/darpa_subt_worlds` (revision `a15a110`); its provenance and conversion
gate are recorded in `sources/subt_ltu_a15a110.json`. It has not yet been
registered as a formal evaluation map.

## External sources

- SubT: download an SDF/Gazebo world and convert the relevant collision mesh to
  an OBJ/STL in `converted/`. Record the original world, commit hash, scale and
  license in a new catalog entry. A public MIT-licensed starting point is
  <https://github.com/LTU-RAI/darpa_subt_worlds>.
- FlightBench: put the downloaded scene folder containing `env.ply`,
  `env-surface.ply` and route CSVs in `raw/flightbench/`. A PLY point cloud is
  not itself a collision volume; create a cleaned voxel/mesh collision asset
  before adding a formal benchmark entry. See <https://thu-uav.github.io/FlightBench/>.
- SensorSimulator: use it to create many training seeds, not as the sealed
  external test set. Its source is <https://github.com/TJU-Aerial-Robotics/SensorSimulator>.

`mesh` entries in the catalog load OBJ/STL through PyBullet, but their route
clearance is deliberately marked unverified until a matching occupancy or
primitive representation is supplied.
