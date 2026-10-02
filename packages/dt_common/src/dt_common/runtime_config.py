# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Resolve portable JSON launch profiles while keeping CLI overrides explicit."""

import argparse
import json
from pathlib import Path


def configure_profile_paths(parser, names):
    """The owning app declares which parser destinations represent paths."""
    names = frozenset(names)
    unknown = names - {action.dest for action in parser._actions}
    if unknown:
        raise ValueError("path options missing from parser: " + ", ".join(sorted(unknown)))
    parser._dt_profile_paths = names
    return parser


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
        # Generated profiles can retain the source profile's path base while
        # snapshotting its arguments. Apps, not dt_common, name path options.
        path_base = (path.parent / document.get("path_base", ".")).resolve()
        path_options = getattr(parser, "_dt_profile_paths", frozenset())
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
            if key in path_options:
                resolved = Path(value).expanduser()
                defaults[key] = str((path_base / resolved).resolve())
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
