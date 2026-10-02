# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Offline scene playback; parsing/timing helpers also work without Isaac."""

try:
    from .extension import ScenePlayerExtension
except ModuleNotFoundError as exc:
    if exc.name != "omni":
        raise
    ScenePlayerExtension = None
