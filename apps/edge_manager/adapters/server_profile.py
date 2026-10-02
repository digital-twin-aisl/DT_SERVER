# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Translate a region run into the server worker's versioned launch profile."""

from copy import deepcopy
from pathlib import Path
from dt_common.infrastructure.deployment import materialize_deployment
from .json_store import read_json, atomic_json


def write_server_profile(region, directory, calibration, endpoint):
    source = Path(region["server_profile"])
    document = read_json(source)
    arguments = deepcopy(document["arguments"])
    arguments.update(
        deployment=str(
            materialize_deployment(
                region["deployment"], directory / "spatial", calibration
            )
        ),
        zenoh_endpoint=endpoint,
        zenoh_config=None,
        scene_zenoh_config=None,
        edge_ids=[edge["edge_id"] for edge in region["edges"]],
        scene_zenoh_endpoint=endpoint,
        scene_zenoh_topic=region["scene_topic"],
        no_scene_zenoh=False,
        no_zmq=True,
        input_mode="independent",
        input_clock="live",
        no_metrics=True,
        viser_debug_output=None,
        root_trajectory_output=None,
        decision_output=None,
        replay_inputs=None,
    )
    profile = directory / "server.json"
    atomic_json(profile, {
        "schema_version": 1,
        "path_base": str((source.parent / document.get("path_base", ".")).resolve()),
        "arguments": arguments,
    })
    return profile
