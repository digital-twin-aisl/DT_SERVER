# Runtime code ownership

The repository separates application ownership from deployment packaging.

```text
apps/
  edge_client/          Edge-only capture, inference, and publishing
  calibration_worker/   Calibration service and VGGT execution
  server_worker/        Server pose inference and scene output

packages/
  dt_common/            Versioned contracts shared by two or more apps

apps/deployments/       Scene assets and runtime manifests, not Python code
```

`dt_common` owns deterministic calibration preprocessing/transport,
USD-to-VoxelPose camera conversion, Ground geometry, and edge workspace
construction. It must not import any `apps.*` module. Applications may depend
on `dt_common`, but must not import implementation code from another app.

Source checkouts expose `packages/dt_common/src` through `apps/__init__.py`.
Normal installations use the `dt-common` package. The root server requirements
install it in editable mode.

An edge-only release vendors both the shared package and the VGGT-Omega model
code as wheels:

```bash
apps/edge_client/tools/vendor_dependencies.sh
```

Afterward, the `apps/edge_client` directory contains both wheel dependencies
under `wheels/`; `install_edge.sh` installs them before the edge application
requirements. The edge feature encoder imports the installed `vggt_omega`
package and never reaches into `calibration_worker` at runtime. This keeps the
edge artifact self-contained without duplicating shared source or making server
code depend on `edge_client`.
