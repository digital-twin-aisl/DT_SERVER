# Spatial workspace ownership

VoxelPose's V2V network still consumes one dense rectangular tensor per edge,
but that tensor is only a compute container. The physical inference region is
the irregular subset selected by the Ground surface, edge AOI, root-clearance
band, and calibrated multi-camera visibility.

## Inputs and ownership

| Input | Owner |
|---|---|
| Ground/walkable geometry | USD scene asset |
| Camera intrinsics and world extrinsics | calibration result |
| Camera-to-edge assignment and optional edge AOI | deployment manifest |
| Reference voxel spacing and network constraints | pose model config |
| Dense grid transform, shape, Ground samples, validity mask | generated `EdgeWorkspace` |

The deployment manifest is loaded directly; it is not merged into the model's
global configuration. `dt_common.spatial.workspace` resolves all deployment paths,
checks the Ground cache against its USD source, and produces the immutable
runtime objects.

## Workspace construction

1. Use the edge's explicit `polygon_xy_m` AOI when one is authored.
2. Otherwise sample the Ground and keep the largest connected region whose
   nominal root height projects inside at least `min_views` camera images.
3. Use the old four-camera containing rectangle only as an enabled fallback
   when automatic visibility cannot produce a connected region.
4. Fit one minimum-area oriented rectangle around the selected footprint.
5. Set its Z interval from Ground height plus `root_clearance_mm`.
6. Preserve the checkpoint's reference millimetres per voxel and round each
   tensor dimension to a V2V-compatible multiple of four.
7. Precompute a dense validity mask. `ProjectLayer` only projects heatmaps into
   this already-resolved workspace.

The current `scene_0812_2` AOIs were seeded from the former camera boundaries
to preserve deployed coverage during migration. They are now explicit data and
can be edited independently of camera positions.

## Edge/server contract

Edge roots remain USD-world coordinates in millimetres. Each edge payload also
carries the scene context ID and calibration SHA-256. The server compares these
values with the same deployment manifest before using the roots and heatmaps.
The server does not reconstruct or rotate the edge's root grid; it only uses
the shared world calibration for local pose projection and the shared Ground
surface for final foot-clearance validation.
