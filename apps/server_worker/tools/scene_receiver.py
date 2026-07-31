import argparse
import json

import zmq


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive server-worker scene outputs over ZMQ",
    )
    parser.add_argument(
        "--bind",
        default="tcp://*:5555",
        help="ZMQ PULL endpoint to bind",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(args.bind)
    print(f"Waiting for scene outputs on {args.bind}")

    try:
        while True:
            scene = socket.recv_json()
            print(json.dumps(scene, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("Stopping scene receiver")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
