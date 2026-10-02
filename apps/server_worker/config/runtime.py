# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Path-valued launch options owned by this application."""

PROFILE_PATH_OPTIONS = frozenset({
    "deployment",
    "zenoh_config",
    "cfg_focus",
    "pose_config",
    "metrics_dir",
    "scene_zenoh_config",
    "root_trajectory_output",
    "viser_debug_output",
    "replay_inputs",
    "decision_output",
    "scene_recording",
})
