# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Operator CLI. Mutations go through the same manager API used by the GUI."""

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .topology import DEFAULT_CATALOG, PROJECT_ROOT
from .protocol import heartbeat_command as _heartbeat_command  # noqa: F401 - compatibility

DEFAULT_REGISTRY_PATH = os.getenv(
    "EDGE_REGISTRY_PATH", str(PROJECT_ROOT / "apps/edge_manager/data/edges.json")
)


def request_api(args, method, path, body=None):
    headers = {"Accept": "application/json"}
    if os.getenv("DT_MANAGER_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["DT_MANAGER_TOKEN"]
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(
        args.manager_url.rstrip("/") + path,
        method=method,
        headers=headers,
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            error = json.load(exc).get("detail", exc.reason)
        except ValueError:
            error = exc.reason
        raise ValueError(str(error)) from exc
    except URLError as exc:
        raise ValueError(
            f"manager unavailable at {args.manager_url}; start/install the manager service: {exc.reason}"
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Digital Twin region and device control"
    )
    parser.add_argument(
        "--manager-url", default=os.getenv("DT_MANAGER_URL", "http://127.0.0.1:8001")
    )
    parser.add_argument(
        "--endpoint", default=os.getenv("ZENOH_ENDPOINT", "tcp/127.0.0.1:7447")
    )
    parser.add_argument("--zenoh-config")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--topic-root", default=os.getenv("EDGE_TOPIC_ROOT", "dt/edges")
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--catalog", default=os.getenv("DT_REGION_CATALOG", str(DEFAULT_CATALOG))
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    server = sub.add_parser(
        "serve", help="Run the persistent management service (one-time installation)"
    )
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8001)
    server.add_argument("--offline-after", type=float, default=15)
    server.add_argument("--heartbeat-interval", type=float, default=5)
    server.add_argument(
        "--data-dir",
        default=os.getenv("DT_DATA_DIR", str(PROJECT_ROOT / "data/manager")),
    )
    listed = sub.add_parser(
        "list", help="Discovered edges, including pending registrations"
    )
    listed.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    for action in ("approve", "revoke", "remove"):
        command = sub.add_parser(action)
        command.add_argument("edge_id")
        if action == "approve":
            command.add_argument("--name")
            command.add_argument("--edge-endpoint")
    command = sub.add_parser("command")
    command.add_argument("edge_id")
    command.add_argument("command", choices=("ping", "info", "shutdown"))
    region = sub.add_parser("region", help="Control a region as one system")
    actions = region.add_subparsers(dest="action", required=True)
    actions.add_parser("list").add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS
    )
    for action in (
        "status",
        "start",
        "stop",
        "logs",
        "recordings",
        "replay",
        "calibrate",
        "register",
        "remove",
    ):
        cmd = actions.add_parser(action)
        cmd.add_argument("region_id")
        cmd.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if action == "start":
            cmd.add_argument(
                "--record",
                action="store_true",
                help="Persist scene JSONL (includes person IDs and positions)",
            )
        elif action == "logs":
            cmd.add_argument("--edge-id")
        elif action == "replay":
            cmd.add_argument("recording_id")
        elif action == "calibrate":
            cmd.add_argument("--edge-id", required=True)
            cmd.add_argument("--reference-video", required=True)
            cmd.add_argument("--marker-tree", required=True)
            cmd.add_argument("--checkpoint", required=True)
        elif action == "register":
            cmd.add_argument("--name", required=True)
            cmd.add_argument(
                "--deployment",
                required=True,
                help="Path on the manager host, relative to catalog",
            )
            cmd.add_argument("--server-profile", required=True)
            cmd.add_argument("--scene-topic")
    return parser


def print_result(result, as_json=False):
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif isinstance(result, dict) and set(result) == {"text"}:
        print(result["text"])
    elif isinstance(result, dict) and "region_id" in result and "edges" in result:
        print(
            f"{result['name']} ({result['region_id']})  [{result.get('state', 'configured')}]"
        )
        if result.get("run_id"):
            print(f"  작업: {result.get('operation')}  run={result['run_id']}")
        for edge in result["edges"]:
            print(
                f"  {edge['edge_id']}: agent={'online' if edge.get('online') else 'offline'}"
                f"  approved={'yes' if edge.get('approved') else 'no'}"
                f"  data={'live' if edge.get('data_fresh') else 'waiting'}"
            )
            print(
                "    cameras: "
                + ", ".join(camera["camera_id"] for camera in edge["cameras"])
            )
            if edge.get("error"):
                print("    " + edge["error"])
        for error in (
            result.get("error"),
            (result.get("worker") or {}).get("error"),
            (result.get("recording_status") or {}).get("error"),
        ):
            if error:
                print("  오류: " + error)
    elif isinstance(result, list):
        if not result:
            print("등록된 항목이 없습니다.")
        for entry in result:
            if "scene_bytes" in entry:
                print(
                    f"{entry['run_id']}  {entry['operation']:<12} {entry['status']:<12} {entry['scene_bytes']} bytes"
                )
            elif "region_id" in entry:
                cameras = sum(len(e["cameras"]) for e in entry["edges"])
                print(
                    f"{entry['region_id']:<24} {entry['state']:<12} {entry['name']}  edges={len(entry['edges'])} cameras={cameras}"
                )
            else:
                print(
                    f"{entry['edge_id']:<24} {'online' if entry.get('online') else 'offline':<8} approved={entry.get('approved', False)}"
                )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.subcommand == "serve":
            from .registry import EdgeRegistry
            from .service import serve

            serve(args, EdgeRegistry(args.registry))
            return
        if args.subcommand == "list":
            result = request_api(args, "GET", "/edges")
        elif args.subcommand in {"approve", "revoke", "remove", "command"}:
            action = args.command if args.subcommand == "command" else args.subcommand
            body = {}
            if action == "approve":
                body = {
                    k: v
                    for k, v in {
                        "name": args.name,
                        "endpoint": args.edge_endpoint,
                    }.items()
                    if v
                }
            result = request_api(
                args, "POST", f"/edges/{quote(args.edge_id, safe='')}/{action}", body
            )
        else:
            action = args.action
            path = (
                "/regions"
                if action == "list"
                else "/regions/" + quote(args.region_id, safe="")
            )
            method, body = "GET", None
            if action in {"start", "stop", "replay", "calibrate"}:
                method, path, body = "POST", path + "/" + action, {}
                if action == "start":
                    body = {"record": args.record}
                elif action == "replay":
                    body = {"recording_id": args.recording_id}
                elif action == "calibrate":
                    body = {
                        k: getattr(args, k)
                        for k in (
                            "edge_id",
                            "reference_video",
                            "marker_tree",
                            "checkpoint",
                        )
                    }
            elif action in {"logs", "recordings"}:
                path += "/" + action
                if action == "logs" and args.edge_id:
                    path += "?" + urlencode({"edge_id": args.edge_id})
            elif action == "register":
                method = "PUT"
                body = {
                    k: getattr(args, k)
                    for k in ("name", "deployment", "server_profile")
                }
                if args.scene_topic:
                    body["scene_topic"] = args.scene_topic
            elif action == "remove":
                method = "DELETE"
            result = request_api(args, method, path, body)
        print_result(result, args.json)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
