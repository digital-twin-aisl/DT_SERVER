import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any
from uuid import uuid4

import zenoh

from .protocol import (
    DEFAULT_TOPIC_ROOT,
    SCHEMA_VERSION,
    EdgeTopics,
    decode_json,
    encode_json,
    make_zenoh_config,
    validate_edge_id,
)
from .registry import EdgeRegistry


DEFAULT_REGISTRY_PATH = os.getenv(
    "EDGE_REGISTRY_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "edges.json"),
)


def _print_record(record: dict[str, Any]) -> None:
    status = record.get("last_status") or {}
    state = "ONLINE" if record.get("online") else "OFFLINE"
    approved = "yes" if record.get("approved") else "no"
    print(
        f"{record['edge_id']:<24} {state:<7} approved={approved:<3} "
        f"host={status.get('hostname', '-')}",
    )


def serve(args: argparse.Namespace, registry: EdgeRegistry) -> None:
    stop_event = threading.Event()

    def stop(*_: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def on_status(sample: Any) -> None:
        try:
            message = decode_json(sample.payload)
            if message.get("kind") != "status":
                raise ValueError("unexpected message kind")
            edge_id = validate_edge_id(str(message["edge_id"]))
            expected_topic = EdgeTopics(edge_id, args.topic_root).status
            if str(sample.key_expr) != expected_topic:
                raise ValueError("status topic and edge_id do not match")
            created, came_online, _ = registry.observe_status(edge_id, message)
            if created:
                print(f"Discovered pending edge: {edge_id}", flush=True)
            elif came_online:
                print(f"Edge is online: {edge_id}", flush=True)
        except Exception as exc:
            print(f"Invalid status message: {exc}", file=sys.stderr, flush=True)

    def on_ack(sample: Any) -> None:
        try:
            message = decode_json(sample.payload)
            if message.get("kind") not in {"command_ack", "config_ack"}:
                raise ValueError("unexpected message kind")
            edge_id = validate_edge_id(str(message["edge_id"]))
            expected_topic = EdgeTopics(edge_id, args.topic_root).ack
            if str(sample.key_expr) != expected_topic:
                raise ValueError("ACK topic and edge_id do not match")
            registry.record_ack(edge_id, message)
            print(
                f"ACK edge={edge_id} command={message.get('command')} "
                f"success={message.get('success')}",
                flush=True,
            )
        except Exception as exc:
            print(f"Invalid ACK message: {exc}", file=sys.stderr, flush=True)

    registry.mark_all_offline()
    config = make_zenoh_config(args.endpoint, args.zenoh_config)
    status_selector = f"{args.topic_root.strip('/')}/*/status"
    ack_selector = f"{args.topic_root.strip('/')}/*/ack"
    with zenoh.open(config) as session:
        status_subscriber = session.declare_subscriber(status_selector, on_status)
        ack_subscriber = session.declare_subscriber(ack_selector, on_ack)
        print(
            f"Edge manager started: status={status_selector} "
            f"registry={registry.path}",
            flush=True,
        )
        while not stop_event.wait(1.0):
            for edge_id in registry.mark_offline(args.offline_after):
                print(f"Edge is offline: {edge_id}", flush=True)
        _ = status_subscriber, ack_subscriber


def list_edges(args: argparse.Namespace, registry: EdgeRegistry) -> None:
    edges = registry.snapshot()["edges"]
    if args.json:
        print(json.dumps(list(edges.values()), ensure_ascii=False, indent=2))
        return
    if not edges:
        print("No edges discovered")
        return
    for edge_id in sorted(edges):
        _print_record(edges[edge_id])


def update_edge(
    registry: EdgeRegistry,
    edge_id: str,
    **fields: Any,
) -> None:
    try:
        record = registry.update(validate_edge_id(edge_id), **fields)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    _print_record(record)


def approve_edge(args: argparse.Namespace, registry: EdgeRegistry) -> None:
    edge_id = validate_edge_id(args.edge_id)
    record = registry.snapshot()["edges"].get(edge_id)
    if record is None:
        raise SystemExit(f"unknown edge: {edge_id}")

    status = record.get("last_status") or {}
    edge_endpoint = (
        args.edge_endpoint
        or status.get("zenoh_endpoint")
        or args.endpoint
    )
    if not edge_endpoint:
        raise SystemExit(
            "edge endpoint is unknown; pass --edge-endpoint or start the edge "
            "agent with --endpoint"
        )

    display_name = args.name or record.get("display_name") or edge_id
    topics = EdgeTopics(edge_id, args.topic_root)
    config_id = uuid4().hex
    message = {
        "schema_version": SCHEMA_VERSION,
        "kind": "edge_config",
        "edge_id": edge_id,
        "config_id": config_id,
        "sent_at": time.time(),
        "data": {
            "approved": True,
            "display_name": display_name,
            "zenoh_endpoint": edge_endpoint,
            "topic_root": topics.root,
            "topics": {
                "status": topics.status,
                "inference": topics.inference,
                "command": topics.command,
                "config": topics.config,
                "ack": topics.ack,
            },
        },
    }
    received = threading.Event()
    response: dict[str, Any] = {}

    def on_ack(sample: Any) -> None:
        try:
            candidate = decode_json(sample.payload)
            if (
                candidate.get("kind") == "config_ack"
                and candidate.get("config_id") == config_id
            ):
                response.update(candidate)
                received.set()
        except Exception:
            return

    config = make_zenoh_config(args.endpoint, args.zenoh_config)
    with zenoh.open(config) as session:
        subscriber = session.declare_subscriber(topics.ack, on_ack)
        session.put(topics.config, encode_json(message))
        if not received.wait(args.timeout):
            raise SystemExit(
                f"config sent but ACK timed out after {args.timeout:.1f}s: {config_id}"
            )
        _ = subscriber

    registry.record_ack(edge_id, response)
    if not response.get("success"):
        raise SystemExit(f"edge rejected configuration: {response.get('error')}")
    update_edge(
        registry,
        edge_id,
        approved=True,
        display_name=display_name,
    )


def send_command(args: argparse.Namespace, registry: EdgeRegistry) -> None:
    edge_id = validate_edge_id(args.edge_id)
    edges = registry.snapshot()["edges"]
    if edge_id not in edges:
        raise SystemExit(f"unknown edge: {edge_id}")
    if not edges[edge_id].get("approved"):
        raise SystemExit(f"edge is not approved: {edge_id}")

    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --parameters JSON: {exc}") from exc
    if not isinstance(parameters, dict):
        raise SystemExit("--parameters must be a JSON object")

    command_id = uuid4().hex
    topics = EdgeTopics(edge_id, args.topic_root)
    message = {
        "schema_version": SCHEMA_VERSION,
        "kind": "command",
        "edge_id": edge_id,
        "command_id": command_id,
        "command": args.command,
        "sent_at": time.time(),
        "parameters": parameters,
    }
    received = threading.Event()
    response: dict[str, Any] = {}

    def on_ack(sample: Any) -> None:
        try:
            candidate = decode_json(sample.payload)
            if candidate.get("command_id") == command_id:
                response.update(candidate)
                received.set()
        except Exception:
            return

    config = make_zenoh_config(args.endpoint, args.zenoh_config)
    with zenoh.open(config) as session:
        subscriber = session.declare_subscriber(topics.ack, on_ack)
        session.put(topics.command, encode_json(message))
        if not received.wait(args.timeout):
            raise SystemExit(
                f"command sent but ACK timed out after {args.timeout:.1f}s: {command_id}"
            )
        _ = subscriber

    registry.record_ack(edge_id, response)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    if not response.get("success"):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Zenoh edge manager")
    parser.add_argument("--endpoint", default=os.getenv("ZENOH_ENDPOINT"))
    parser.add_argument("--zenoh-config")
    parser.add_argument(
        "--topic-root",
        default=os.getenv("EDGE_TOPIC_ROOT", DEFAULT_TOPIC_ROOT),
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--offline-after", type=float, default=15.0)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--json", action="store_true")

    approve = subparsers.add_parser("approve")
    approve.add_argument("edge_id")
    approve.add_argument("--name")
    approve.add_argument("--edge-endpoint")
    approve.add_argument("--timeout", type=float, default=5.0)

    for name in ("revoke", "remove"):
        command = subparsers.add_parser(name)
        command.add_argument("edge_id")

    command = subparsers.add_parser("command")
    command.add_argument("edge_id")
    command.add_argument("command", choices=("ping", "info", "shutdown"))
    command.add_argument("--parameters", default="{}")
    command.add_argument("--timeout", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = EdgeRegistry(args.registry)
    try:
        if args.subcommand == "serve":
            if args.offline_after <= 0:
                raise SystemExit("--offline-after must be greater than zero")
            serve(args, registry)
        elif args.subcommand == "list":
            list_edges(args, registry)
        elif args.subcommand == "approve":
            if args.timeout <= 0:
                raise SystemExit("--timeout must be greater than zero")
            approve_edge(args, registry)
        elif args.subcommand == "revoke":
            update_edge(registry, args.edge_id, approved=False)
        elif args.subcommand == "remove":
            registry.remove(validate_edge_id(args.edge_id))
            print(f"Removed edge: {args.edge_id}")
        elif args.subcommand == "command":
            if args.timeout <= 0:
                raise SystemExit("--timeout must be greater than zero")
            send_command(args, registry)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
