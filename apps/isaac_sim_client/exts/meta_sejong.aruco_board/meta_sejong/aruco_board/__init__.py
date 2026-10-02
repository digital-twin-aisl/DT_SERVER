# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Isaac Sim extension package.

The guarded import keeps the reusable generator importable in a plain USD Python
environment, where the Kit-only ``omni`` package is intentionally unavailable.
"""

try:
    from .extension import ArucoBoardExtension
except ModuleNotFoundError as exc:
    if exc.name != "omni":
        raise
    ArucoBoardExtension = None

__all__ = ["ArucoBoardExtension"]
