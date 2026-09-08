import json
import logging
from dt_common.zenoh_transport import LatestPublisher


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
