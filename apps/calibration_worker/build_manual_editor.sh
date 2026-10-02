#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
viewer_dir="$task_dir/../frontend_api/web"
if [[ ! -x "$viewer_dir/node_modules/.bin/esbuild" ]]; then
  echo "Missing existing viewer dependencies. Run: cd apps/frontend_api/web && npm ci" >&2
  exit 1
fi
"$viewer_dir/node_modules/.bin/esbuild" "$task_dir/manual_web/editor.js" \
  --bundle --format=esm --minify --alias:three="$viewer_dir/node_modules/three/build/three.module.js" \
  --alias:three/addons="$viewer_dir/node_modules/three/examples/jsm" \
  --outfile="$task_dir/manual_web/dist/editor.js"
