#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"
map_python="${MAP_PYTHON:-python3}"
map_usd="${VIEWER_USD_PATH:-$repo_dir/All/2025_SejongUniv_All.usd}"
map_deployment="${VIEWER_DEPLOYMENT:-$repo_dir/apps/deployments/scene_0812_poc.json}"
map_output="$repo_dir/apps/frontend_api/assets"
# Export to a staging directory; keep the served pair intact if conversion fails.
map_staging="$(mktemp -d "$map_output/.export-XXXXXX")"
trap 'rm -rf "$map_staging"' EXIT
"$map_python" apps/frontend_api/tools/export_map.py --usd "$map_usd" --deployment "$map_deployment" --output "$map_staging" "$@"
apps/frontend_api/web/node_modules/.bin/gltf-transform optimize "$map_staging/map.glb" "$map_staging/map.optimized.glb" \
  --compress meshopt --simplify false --texture-compress webp --texture-size 1024
"$map_python" - "$map_staging" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
path = folder / 'map.json'
metadata = json.loads(path.read_text())
metadata['uncompressed_bytes'] = metadata['bytes']
metadata['bytes'] = (folder / 'map.optimized.glb').stat().st_size
metadata['compression'] = 'meshopt + WebP; geometry simplification disabled'
path.write_text(json.dumps(metadata, indent=2) + '\n')
PY
mv "$map_staging/map.optimized.glb" "$map_output/map.glb"
mv "$map_staging/map.json" "$map_output/map.json"
