# dt-common

Shared, CPU-side contracts used by more than one DT runtime:

- deterministic calibration preprocessing and feature transport;
- USD-world calibration conversion for VoxelPose;
- Ground surface loading/cache validation;
- edge workspace construction and identity hashing.

Application-specific inference, transport clients, model code, and device
management do not belong in this package. Runtime applications provide NumPy,
OpenCV, and optional OpenUSD dependencies appropriate for their platform.

For local server development, install it from the repository root:

```bash
python -m pip install -e packages/dt_common
```

For an edge-only deployment bundle, vendor the required wheels into
`apps/edge_client`:

```bash
apps/edge_client/tools/vendor_dependencies.sh
```

The command also vendors the edge calibration encoder's VGGT-Omega wheel.
`apps/edge_client/install_edge.sh` prefers the bundled wheels and falls back to
source packages when it is running from a complete repository checkout.
