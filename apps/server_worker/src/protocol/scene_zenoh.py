# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
import logging
from dt_common.infrastructure.zenoh import LatestPublisher


logger = logging.getLogger(__name__)
DEFAULT_SCENE_TOPIC = "meta-sejong/scene/v1"


def encode_scene(scene):
    return json.dumps(scene, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


class ZenohScenePublisher(LatestPublisher):
    def __init__(self, topic=DEFAULT_SCENE_TOPIC, **kwargs):
        super().__init__(topic, **kwargs)

    def send_scene(self, scene):
        return self.send_payload(encode_scene(scene))

    def send_payload(self, payload):
        try:
            self.send(payload)
            return True
        except RuntimeError:
            logger.exception("Cannot publish scene")
            return False
