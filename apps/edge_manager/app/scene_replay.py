# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Replay stored scene records through the normal scene consumer interface."""

import argparse
import json
import math
import time
import zenoh
from dt_common.infrastructure.zenoh import make_zenoh_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--topic", required=True)
    args = parser.parse_args()
    previous = None
    with (
        zenoh.open(make_zenoh_config(args.endpoint)) as session,
        open(args.input) as stream,
    ):
        for line in stream:
            scene = json.loads(line)
            stamp = float(scene["timestamp"])
            if not math.isfinite(stamp) or scene.get("schema_version") != 1:
                raise ValueError("invalid recorded scene")
            if previous is not None:
                time.sleep(min(5, max(0, stamp - previous)))
            previous = stamp
            scene.setdefault("runtime", {})["playback"] = True
            session.put(args.topic, json.dumps(scene, allow_nan=False).encode())


if __name__ == "__main__":
    main()
