# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A distributed digital twin for a university campus. Jetson edge devices run 2D perception (heatmaps, root, ReID) on their RTSP cameras. A GPU server joins those observations into one 3D scene of people (global ID, root position, optional 15-joint pose). The scene is published to a browser viewer and to Isaac Sim. The unit of operation is a **region** (구역). The current region is `center-b1-corridor`: `edge_1` has cameras 2/4/6/8 and `edge_2` has cameras 1/3/5/7. User-facing docs (README, ARCHITECTURE, `docs/guide/`, per-app READMEs) are written in Korean.

All transport uses **Zenoh 1.9.0**; the router defaults to `tcp/127.0.0.1:7447`. The old gRPC→Redis→dl_worker→sim_backend design no longer exists, so ignore any references to it.

## Layout and ownership

- `packages/dt_common` (`dt-common`, v0.3.0, `src/` layout): CPU-only contracts shared by every host. It covers the edge wire contract and topics (`contracts/edge.py`), the `SceneOutput` contract (`contracts/scene.py`), the ZNH2 inference codec, calibration preprocessing, ground/workspace geometry, per-run deployment snapshots and `ManagedProcess`. Model code and device policy do not belong here. The manager path expects every host to run the same dt-common version.
- `apps/edge_manager`: a single long-running service. It runs the Zenoh discovery/heartbeat loop, the FastAPI management API (default `127.0.0.1:8001`) and the CLI. `ControlPlane` (`app/control.py`) is the only owner of region runs: validate → start server + edges → reconcile → stop. `EdgeRegistry` (`app/registry.py`) persists discovered and approved devices. `RegionCatalog` (`app/topology.py`) reads `apps/deployments/regions.json`.
- `apps/edge_client`: `agent.py` is the always-on edge agent. It handles identity, the command/ACK/lease protocol and camera status, and supervises `inference.py` (the actual edge inference) through `application/runtime.py:EdgeInferenceRuntime`.
- `apps/server_worker`: `inference.py` is the server inference loop. Its pipeline is input contract/freshness validation → global ID/root association → global Rank/LOD every 0.5 s → 3D pose only for LOD-2 objects → TTL scene merge → ground-height filter → `SceneOutput`. Pure logic lives in `domain/`. Model code is in `src/pose`, and transport (zenoh/zmq/replay/sync) is in `src/protocol`.
- `apps/calibration_worker`: VGGT-Omega + ArUco marker-tree camera calibration (`inference.py`), plus a manual calibration editor (`manual_editor.py` + `manual_web/`).
- `apps/frontend_api`: FastAPI GUI on port 8005. It relays the manager API from the same origin (`app/control_proxy.py`) and holds no control logic of its own. It also serves a Three.js scene viewer that is built from `web/`.
- `apps/isaac_sim_client`: Isaac Sim scene subscriber and extensions. It must run under Isaac Sim's own `python.sh` (omni/pxr ABI), not the server venv.
- `apps/deployments`: the region catalog, the PoC manifest `scene_0812_poc.json` (cameras, order, AOI, calibration/ground references) and the per-host launch profiles in `poc/`.
- Compatibility shims: top-level modules such as `apps/server_worker/{scene_state,root_tracks,lod_scheduler,priority_engine}.py`, `apps/edge_client/runtime.py` and `dt_common/{process,deployment,zenoh_transport}.py` only re-export from the `domain/`, `application/` or `infrastructure/` modules. Edit the real module, not the shim.
- Not part of the main tree: `new_DT_SERVER/` (an untracked, separate redesign with its own pyproject/venv and no `apps.*` imports), plus `Faster-VoxelPose/` and `SelfPose3d/` (untracked reference upstreams). Git submodules: `apps/calibration_worker/vggt-omega` and `apps/edge_client/src/reid/fast-reid`.

## Cross-cutting contracts (read several files before changing these)

- **Zenoh keys**: edge topics are `dt/edges/{edge_id}/{status,inference,command,config,ack,cameras}` (`EdgeTopics`). The scene topic for this region is `meta-sejong/scene/v1`. Camera keys are `{edge_id}/camera/{n}`. Edge IDs do not encode region membership. The manager refuses to start if its `--topic-root` differs from any region's topic root.
- **ZNH2 inference packets**: JSON metadata plus a restricted-dtype ndarray buffer, compressed with zstd, never pickle. The receiver checks camera order, the effective calibration digest, scene/workspace ID, tensor sizes, clock type and session/sequence. ZNH1 is rejected, so codec changes require updating the edge and the server together.
- **Units and frames**: model roots and `SceneOutput` use **millimetres** in the USD world frame (Z-up). Priority/hazard inputs use metres. The Isaac stage uses `metersPerUnit=0.01`.
- **Run state is evidence-based**: process spawn, agent heartbeat and fresh `SceneOutput.runtime.input_status` are separate signals. The state is `running` only when all three hold. Replayed scenes carry `runtime.playback=true` and must not count as live observation. Edges stop managed inference after 30 s without a lease renewal. Processes retry at most 3 times with exponential backoff. A manager restart marks unfinished runs as `interrupted` and never resumes them.
- **Calibration adoption**: a calibration finished through the manager is written to `calibration-overrides.json` only after the edge ACKs it and its coordinate frame matches. On the next start, only that edge's cameras are merged in. Original deployment and calibration files are never modified.
- Per-run artifacts go to `data/manager/runs/<run_id>/` (manifest, generated server config, logs, optional `scenes.jsonl`). Recording is opt-in and bounded (`DT_RECORDING_MAX_BYTES`, default 1 GiB).
- RTSP URLs and credentials exist only in edge-local files (`apps/edge_client/config/cameras.local.yaml`, `edge.local.json`). Never commit them.

## Commands

Run all commands from the repo root. Modules are run as `python -m apps.<pkg>`, so the repo root must be on `sys.path`.

```bash
# Server env setup (Python 3.10/3.11, CUDA torch installed first; see docs/guide/02-install-server.md)
python -m pip install -r requirements.txt                 # also installs packages/dt_common editable
python -m pip install -r apps/edge_manager/requirements.txt -r apps/frontend_api/requirements.txt
npm ci --prefix apps/frontend_api/web && npm run build --prefix apps/frontend_api/web   # -> app/static/dist/viewer.js
apps/calibration_worker/build_manual_editor.sh            # bundles manual_web using frontend_api/web's node_modules

# Dev stack (three processes)
zenohd -c apps/edge_manager/config/zenoh-router-poc.json5
python -m apps.edge_manager serve
python -m uvicorn apps.frontend_api.app.main:app --host 127.0.0.1 --port 8005

# Operations go through the manager, not individual inference scripts
python -m apps.edge_manager list | approve edge_1
python -m apps.edge_manager region start|status|stop|logs center-b1-corridor [--record] [--json]
```

Running `apps/server_worker/inference.py` or the edge `inference.py` directly is for diagnostics and research only (see `apps/server_worker/docs/poc-runtime.md`). Never run them alongside a managed region. `apps/server_worker/tools/scene_zenoh_subscriber.py` prints the published scenes.

### Tests

Python tests are `unittest`-style files. Each one inserts the repo root and `packages/dt_common/src` into `sys.path` itself, so they run under either pytest or unittest without installing anything. Most need the server env (numpy, torch, zenoh).

```bash
python -m pytest apps/server_worker/tests apps/frontend_api/tests apps/calibration_worker/tests apps/deployments/rootnet_v2/tests
python -m pytest apps/server_worker/tests/test_ground_filter.py -k <name>        # single test
python -m unittest apps/server_worker/tests/test_ground_filter.py

npm test --prefix apps/frontend_api/web                 # node:test (playback, recording controls)
npm run test:browser --prefix apps/frontend_api/web     # browser smoke test
node --test apps/calibration_worker/manual_web/geometry.test.js   # needs `three` resolvable (frontend_api/web node_modules)
python -m pytest "apps/isaac_sim_client/exts/meta_sejong.scene_player/tests"
```

Lint with `ruff check apps packages examples tools` (config in `ruff.toml`, submodules excluded). Dev tools are in `requirements-dev.txt`.

## Open-source release

- License: LGPL-2.1-or-later, REUSE-compliant (`reuse lint`). New source files need the SPDX header used across the tree. VoxelPose-derived model files stay MIT. `apps/calibration_worker/patches/*.patch` is FAIR Noncommercial (annotated in `REUSE.toml`).
- Ultralytics (AGPL) and VGGT-Omega (FAIR NC) are optional and must never become hard dependencies. Weights, media and recordings never go in git.
- The public release is the orphan branch `release/oss`, produced from committed dev HEAD by `python release/export.py` (see `release/README.md`). The dev branch is the source of truth, and `release/` plus `CLAUDE.md` are excluded from the export.
- `python tools/scan_sensitive.py .` must report 0 findings. Leaked literals live in the git-ignored `data/local/release-denylist.txt`.
