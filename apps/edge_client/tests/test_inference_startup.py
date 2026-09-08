import json
from types import SimpleNamespace

import numpy as np

from apps.edge_client.inference import count_valid_roots, print_startup_metadata
from apps.edge_client.src.protocol import zenoh as zenoh_protocol


class _SpatialContext:
    edge_camera_ids = {"edge_1": (2, 4, 6, 8)}

    @staticmethod
    def identity():
        return {"scene_id": "scene-test"}


def test_startup_metadata_contains_resolved_edge_information(capsys):
    args = SimpleNamespace(
        edge_id_file="/tmp/edge.local.json",
        dataset=False,
        deployment="/tmp/scene.json",
        no_reid=False,
        no_zenoh=False,
    )

    print_startup_metadata(
        edge_id="edge_1",
        edge_metadata={
            "display_name": "dt_jetson_1",
            "approved": True,
            "camera_url": "rtsp://user:secret@example.test/live",
        },
        args=args,
        spatial_context=_SpatialContext(),
        live_cameras=None,
        use_tensorrt=True,
        zenoh_endpoint="udp/server.example.test:10020?rel=1",
        inference_topic="dt/edges/edge_1/inference",
    )

    output = capsys.readouterr().out
    metadata = json.loads(output.split("\n", 1)[1])
    assert metadata["edge_id"] == "edge_1"
    assert metadata["display_name"] == "dt_jetson_1"
    assert metadata["camera_ids"] == [2, 4, 6, 8]
    assert metadata["server_router"] == {
        "endpoint": "udp/server.example.test:10020?rel=1",
        "topic": "dt/edges/edge_1/inference",
        "status": "connecting",
    }
    assert "secret" not in output


def test_zenoh_sender_exposes_router_connection(monkeypatch):
    class FakeInfo:
        @staticmethod
        def routers_zid():
            return ("router-1",)

        @staticmethod
        def links():
            return ("link-1",)

    class FakePublisher:
        @staticmethod
        def put(payload):
            del payload

    class FakeSession:
        info = FakeInfo()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        @staticmethod
        def declare_publisher(topic, **kwargs):
            del topic
            assert kwargs["congestion_control"] == zenoh_protocol.zenoh.CongestionControl.DROP
            return FakePublisher()

    monkeypatch.setattr(zenoh_protocol.zenoh, "open", lambda config: FakeSession())

    sender = zenoh_protocol.ZenohSender(topic="dt/edges/edge_1/inference")
    try:
        assert sender.router_connected is True
        assert sender.router_ids == ("router-1",)
        assert sender.link_count == 1
    finally:
        sender.close()


def test_valid_root_count_matches_server_candidate_rule():
    roots = np.array(
        [
            [
                [100.0, 200.0, 300.0, 0.0, 0.9],
                [400.0, 500.0, 600.0, 1.0, 0.7],
                [0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, np.nan],
            ]
        ],
        dtype=np.float32,
    )

    assert count_valid_roots(roots) == 2
