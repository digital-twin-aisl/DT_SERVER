# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Compatibility facade. Identity storage and shared messages have separate owners."""

from dt_common.contracts.edge import (  # noqa: F401
    DEFAULT_TOPIC_ROOT, EDGE_ID_PATTERN, SCHEMA_VERSION, EdgeTopics,
    decode_json, encode_json, validate_edge_id,
)
from apps.edge_client.infrastructure.identity import (  # noqa: F401
    DEFAULT_IDENTITY_PATH, load_or_create_edge_id, load_edge_metadata, save_edge_metadata,
)
