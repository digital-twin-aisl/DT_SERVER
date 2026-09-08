"""Resolve portable JSON launch profiles while keeping CLI overrides explicit."""

import argparse
import json
from pathlib import Path


PATH_OPTIONS = {
    "deployment",
    "camera_config",
    "edge_id_file",
    "zenoh_config",
    "cfg_focus",
    "pose_config",
    "example_folder",
    "metrics_dir",
    "scene_zenoh_config",
    "root_trajectory_output",
    "viser_debug_output",
    "record_inputs",
    "replay_inputs",
    "record_output",
    "decision_output",
}


def parse_runtime_args(parser, argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--runtime-config")
    initial, _ = bootstrap.parse_known_args(argv)
    if initial.runtime_config:
        path = Path(initial.runtime_config).expanduser().resolve()
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            parser.error("runtime config requires schema_version=1")
        defaults = dict(document.get("arguments", {}))
        actions = {action.dest: action for action in parser._actions}
        unknown = set(defaults) - set(actions)
        if unknown:
            parser.error("unknown runtime options: " + ", ".join(sorted(unknown)))
        for key, value in defaults.items():
            action = actions[key]
            if value is None:
                continue
            if (
                isinstance(
                    action,
                    (
                        argparse._StoreTrueAction,
                        argparse._StoreFalseAction,
                        argparse.BooleanOptionalAction,
                    ),
                )
                and type(value) is not bool
            ):
                parser.error(f"{key} must be a JSON boolean")
            if key in PATH_OPTIONS:
                resolved = Path(value).expanduser()
                defaults[key] = str((path.parent / resolved).resolve())
            elif action.type is json.loads and not isinstance(value, str):
                # JSON profiles already contain decoded lists/objects.
                defaults[key] = value
            elif (
                action.type is not None
                and action.nargs is None
                and not isinstance(action, argparse._AppendAction)
            ):
                defaults[key] = action.type(value)
            if action.choices and defaults[key] not in action.choices:
                parser.error(f"invalid {key}: {defaults[key]}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)
