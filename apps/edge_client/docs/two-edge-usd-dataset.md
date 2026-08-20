# Two-edge USD VoxelPose dataset run

The `data_0812_2` recording is split by physical camera number:

- `edge_1`: cameras `2, 4, 6, 8`
- `edge_2`: cameras `1, 3, 5, 7`

Both edges load the same calibration result and express roots in absolute USD
world millimetres. The server uses the same camera ordering and calibration for
pose projection. The shared
`apps/deployments/scene_0812_2.json` manifest binds the Ground asset,
calibration, edge/camera assignments, edge AOIs, and workspace policy in one
place. Each explicit AOI is checked against the calibrated camera frusta and
Ground height band, enclosed by one rotated dense grid, and converted to voxel
counts that preserve the checkpoint's reference mm/voxel. Runtime processes read the portable
`apps/deployments/cache/scene_0812_2_ground.npz` triangle cache, so their Python
environment does not need the OpenUSD `pxr` package. The cache is checked
against the SHA-256 digest of `Ground.usd` to prevent stale geometry.

The camera-position rectangle is no longer the runtime source of truth. The
current manifest AOIs were seeded from the former boundaries for a behavior-safe
migration, but they can now be authored independently. When an edge has no
explicit AOI, the workspace builder uses the largest connected Ground region
visible from at least `workspace.min_views` cameras; the old camera rectangle is
only an optional last-resort fallback.

If `Ground.usd` changes, regenerate the cache once from a Python environment
that has OpenUSD installed:

```bash
cd /home/dojan/DT_SERVER
python -m dt_common.spatial.ground export \
  Ground.usd \
  apps/deployments/cache/scene_0812_2_ground.npz
```

Run all commands from the indicated application directory. A Zenoh router must
already be reachable at the endpoint supplied to all three processes.

## Server

```bash
cd /home/dojan/DT_SERVER/apps/server_worker
python inference.py \
  --deployment ../deployments/scene_0812_2.json \
  --zenoh-endpoint localhost:7447 \
  --buffer-size 120 \
  --sync-tolerance 0.001 \
  --no-zmq
```

## Edge 1

```bash
cd /home/dojan/DT_SERVER/apps/edge_client
python inference.py \
  --dataset \
  --example_folder data/data_0812_2_edge_1 \
  --edge-id edge_1 \
  --edge-id-file config/edge_1.dataset.json \
  --deployment ../deployments/scene_0812_2.json \
  --zenoh-endpoint localhost:7447
```

## Edge 2

```bash
cd /home/dojan/DT_SERVER/apps/edge_client
python inference.py \
  --dataset \
  --example_folder data/data_0812_2_edge_2 \
  --edge-id edge_2 \
  --edge-id-file config/edge_2.dataset.json \
  --deployment ../deployments/scene_0812_2.json \
  --zenoh-endpoint localhost:7447
```

The separate identity files are required when both dataset clients run from one
checkout. Dataset input advances exactly one synchronized video frame after
each inference. Payload timestamps use relative video time, so the terminals do
not need to be started in the same instant. Add `--tensorrt` to each edge command
after the PyTorch path has been verified on the target GPU.

The legacy `data_0705/hdVideos/*.mp4 + calibration/*.pkl` layout remains
supported when `--deployment` is omitted; it uses the original fixed model
space without Ground-aware workspace construction.
