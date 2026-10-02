# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Persistent identity owned only by the edge agent."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4
from dt_common.contracts.edge import validate_edge_id

DEFAULT_IDENTITY_PATH = Path(__file__).resolve().parents[1] / "config" / "edge.local.json"


def load_or_create_edge_id(
    path: str | Path = DEFAULT_IDENTITY_PATH,
    requested_edge_id: str | None = None,
) -> str:
    identity_path = Path(path)
    if identity_path.exists():
        data = load_edge_metadata(identity_path)
        edge_id = validate_edge_id(str(data["edge_id"]))
        if requested_edge_id is not None and requested_edge_id != edge_id:
            raise ValueError(
                f"stored edge_id is {edge_id!r}; refusing requested {requested_edge_id!r}"
            )
        return edge_id

    edge_id = validate_edge_id(requested_edge_id or f"edge-{uuid4().hex[:12]}")
    save_edge_metadata(identity_path, {"edge_id": edge_id})
    return edge_id



def load_edge_metadata(path: str | Path = DEFAULT_IDENTITY_PATH) -> dict[str, Any]:
    identity_path = Path(path)
    data = json.loads(identity_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("edge identity must be a JSON object")
    validate_edge_id(str(data["edge_id"]))
    return data



def save_edge_metadata(path: str | Path, data: dict[str, Any]) -> None:
    identity_path = Path(path)
    validate_edge_id(str(data["edge_id"]))
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=identity_path.parent,
        prefix=f".{identity_path.name}.",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, identity_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

