"""Simple Zenoh subscriber for checking edge inference publications."""

import argparse
import json
from dt_common.inference_codec import decode_frame
import time

import zenoh


def make_config(endpoint=None, config_path=None):
    if config_path:
        return zenoh.Config.from_file(config_path)

    if endpoint:
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

    return zenoh.Config()


def main():
    parser = argparse.ArgumentParser(description="Subscribe to edge inference output")
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument("--topic", default="dt/edges/*/inference")
    parser.add_argument("--config")
    args = parser.parse_args()

    received = 0

    def on_sample(sample):
        nonlocal received
        received += 1

        payload = sample.payload.to_bytes()
        message = None
        try:
            message = decode_frame(payload)
        except Exception as exc:
            print(f"Invalid inference frame: {exc}", flush=True)

        print(
            f"[{received}] topic={sample.key_expr} "
            f"size={len(payload) / 1024**2:.2f} MB",
            flush=True,
        )

        if isinstance(message, dict):
            print(
                f"    fields={list(message.keys())} "
                f"reid={len(message.get('reid', []))} "
                f"time={message.get('time')}",
                flush=True,
            )

            if message.get("reid"):
                first = message["reid"][0]
                print(
                    f"    first person: cam={first.get('cam')} "
                    f"frame={first.get('frame')} "
                    f"person_idx={first.get('person_idx')}",
                    flush=True,
                )
        else:
            print("    payload is not a valid ZNH2 frame", flush=True)

    print(f"Connecting to {args.endpoint}")
    print(f"Subscribing to {args.topic}")

    try:
        with zenoh.open(make_config(args.endpoint, args.config)) as session:
            with session.declare_subscriber(args.topic, on_sample):
                print("Waiting for messages... Press Ctrl+C to stop.", flush=True)
                while True:
                    time.sleep(1)
    except KeyboardInterrupt:
        print(f"Stopped. Received {received} message(s).")


if __name__ == "__main__":
    main()
