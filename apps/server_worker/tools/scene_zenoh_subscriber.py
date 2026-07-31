"""Print SceneOutput messages without starting Isaac Sim."""

import argparse
import json
import time

import zenoh


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:7447")
    parser.add_argument("--topic", default="meta-sejong/scene/v1")
    return parser


def make_config(endpoint: str) -> zenoh.Config:
    if "/" not in endpoint:
        endpoint = f"tcp/{endpoint}"
    return zenoh.Config.from_json5(
        json.dumps(
            {
                "mode": "client",
                "connect": {"endpoints": [endpoint]},
            }
        )
    )


def main() -> None:
    args = get_parser().parse_args()

    def on_sample(sample) -> None:
        try:
            scene = json.loads(sample.payload.to_bytes().decode("utf-8"))
            ids = [person["global_id"] for person in scene.get("people", [])]
            print(
                f"timestamp={scene.get('timestamp')} "
                f"people={len(ids)} global_ids={ids}",
                flush=True,
            )
        except Exception as exc:
            print(f"invalid scene payload: {exc}", flush=True)

    with zenoh.open(make_config(args.endpoint)) as session:
        subscriber = session.declare_subscriber(args.topic, on_sample)
        print(
            f"subscribed: endpoint={args.endpoint} topic={args.topic}",
            flush=True,
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        _ = subscriber


if __name__ == "__main__":
    main()
