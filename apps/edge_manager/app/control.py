# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""One application service for CLI and GUI. No GPU libraries are imported here."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

from dt_common.contracts.scene import OUTPUT_SCHEMA_VERSION
from dt_common.infrastructure.process import ManagedProcess
from apps.edge_manager.domain.calibration import merge_edge_calibration
from dt_common.infrastructure.deployment import (
    read_deployment_calibration,
)
from .topology import PROJECT_ROOT
from apps.edge_manager.adapters.json_store import atomic_json, read_json
from dt_common.contracts.edge import validate_edge_id
from .recording import SceneRecorder
from apps.edge_manager.adapters.server_profile import write_server_profile


class ControlPlane:
    def __init__(
        self,
        catalog,
        registry,
        transport,
        data_dir,
        endpoint,
        *,
        process_factory=ManagedProcess,
    ):
        self.catalog, self.registry, self.transport = catalog, registry, transport
        self.data_dir = Path(data_dir).resolve()
        self.endpoint = endpoint
        self.process_factory = process_factory
        self.lock = threading.RLock()
        self.runs = {}
        self.scenes = {}
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="region")
        self.futures = {}
        self.closing = False
        self.calibration_path = self.data_dir / "calibration-overrides.json"
        self.calibration_overrides = (
            read_json(self.calibration_path) if self.calibration_path.exists() else {}
        )
        # A manager restart never silently resumes a previous live/recording job.
        for path in self.data_dir.glob("runs/*/manifest.json"):
            record = read_json(path)
            if record.get("status") not in {
                "stopped",
                "completed",
                "failed",
                "interrupted",
            }:
                record.update(status="interrupted", ended_at=time.time())
                atomic_json(path, record)

    def _calibration(self, region):
        calibration = read_deployment_calibration(region["deployment"])
        overrides = self.calibration_overrides.get(region["region_id"], {})
        for edge in region["edges"]:
            if edge["edge_id"] in overrides:
                calibration = merge_edge_calibration(
                    calibration,
                    read_json(overrides[edge["edge_id"]]),
                    edge["edge_id"],
                    [c["camera_id"] for c in edge["cameras"]],
                )
        return calibration


    def _new_run(self, region_id, operation, options):
        if self.closing:
            raise ValueError("manager is shutting down")
        region = self.catalog.get(region_id)
        current = self.runs.get(region_id)
        if current and current["desired"]:
            if current["operation"] == operation and current["options"] == options:
                return current, False
            raise ValueError(
                "stop the current region job before starting a different one"
            )
        if current and current["state"] == "stopping":
            raise ValueError("wait for the region to stop")
        run_id = uuid4().hex
        directory = self.data_dir / "runs" / run_id
        directory.mkdir(parents=True)
        runtime = dict(
            run_id=run_id,
            region=region,
            region_id=region_id,
            operation=operation,
            options=options,
            desired=True,
            state="starting",
            phase="validating",
            error=None,
            started_at=time.time(),
            directory=directory,
            worker=self.process_factory(directory / "worker.log", cwd=PROJECT_ROOT),
            edge_errors={},
            edge_results={},
            last_commands={},
            command=None,
            recorder=None,
        )
        self.runs[region_id] = runtime
        self.scenes.pop(region["scene_topic"], None)
        self._persist(runtime)
        return runtime, True

    def _persist(self, run):
        atomic_json(
            run["directory"] / "manifest.json",
            {
                "schema_version": 1,
                "run_id": run["run_id"],
                "region_id": run["region_id"],
                "operation": run["operation"],
                "options": run["options"],
                "status": run["state"],
                "started_at": run["started_at"],
                "updated_at": time.time(),
                "error": run["error"],
                "scene_schema_version": 1,
                "contains": ["global_id", "position", "pose", "timestamp"],
                "privacy_review": "not_asserted",
                "recording": run["recorder"].status() if run["recorder"] else None,
            },
        )

    def start(self, region_id, *, record=False):
        if type(record) is not bool:
            raise ValueError("record must be a boolean")
        with self.lock:
            run, created = self._new_run(region_id, "live", {"record": record})
            if created:
                try:
                    run["calibration"] = self._calibration(run["region"])
                    profile = write_server_profile(
                        run["region"], run["directory"], run["calibration"], self.endpoint
                    )
                    run["command"] = [
                        os.getenv("DT_SERVER_PYTHON", sys.executable),
                        str(PROJECT_ROOT / "apps/server_worker/inference.py"),
                        "--runtime-config",
                        str(profile),
                    ]
                    run["deployment_sha256"] = hashlib.sha256(
                        Path(run["region"]["deployment"]).read_bytes()
                    ).hexdigest()
                    run["worker"].start(
                        run["command"] + ["--validate-only"], restart=False
                    )
                    if record:
                        run["recorder"] = SceneRecorder(
                            run["directory"] / "scenes.jsonl"
                        )
                except Exception as exc:
                    run.update(desired=False, state="failed", error=str(exc))
                    self._persist(run)
                    raise
            return self.status(region_id)

    def stop(self, region_id):
        with self.lock:
            self.catalog.get(region_id)
            run = self.runs.get(region_id)
            if run and (run["desired"] or run["state"] not in {"stopped", "completed"}):
                run.update(desired=False, state="stopping")
                self._persist(run)
            return self.status(region_id)

    def receive_scene(self, topic, payload):
        try:
            if len(payload) > 8 * 1024 * 1024:
                return
            scene = json.loads(payload)
            if scene.get("schema_version") != OUTPUT_SCHEMA_VERSION or not isinstance(
                scene.get("people"), list
            ):
                return
            timestamp = scene.get("timestamp")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
            ):
                return
            runtime = scene.get("runtime") or {}
            if not isinstance(runtime, dict):
                return
            inputs = runtime.get("input_status", {})
            if not isinstance(inputs, dict) or any(
                value not in {"online", "offline"} for value in inputs.values()
            ):
                return
            with self.lock:
                self.scenes[topic] = (
                    time.monotonic(),
                    inputs,
                )
                for run in self.runs.values():
                    if (
                        run["desired"]
                        and run["region"]["scene_topic"] == topic
                        and run["recorder"]
                    ):
                        run["recorder"].put(payload)
        except (ValueError, AttributeError, TypeError):
            return

    def tick(self):
        with self.lock:
            for region_id, run in self.runs.items():
                future = self.futures.get(region_id)
                if future is not None and not future.done():
                    continue
                if run["desired"] or run["state"] == "stopping":
                    self.futures[region_id] = self.pool.submit(self._reconcile, run)

    def _reconcile(self, run):
        try:
            if not run["desired"]:
                for edge in (
                    run["region"]["edges"] if run["operation"] == "live" else []
                ):
                    try:
                        self.transport.request(
                            edge["edge_id"], "stop_inference", {"run_id": run["run_id"]}
                        )
                        run["edge_errors"].pop(edge["edge_id"], None)
                    except Exception as exc:
                        run["edge_errors"][edge["edge_id"]] = str(exc)
                run["worker"].stop()
                if run["recorder"]:
                    run["recorder"].close()
                with self.lock:
                    run["state"] = "stopped"
                    if run["edge_errors"]:
                        run["error"] = (
                            "some edges did not acknowledge stop; their 30-second lease will expire"
                        )
                    self._persist(run)
                return
            run["worker"].tick()
            state = run["worker"].status()
            if state.get("restart_exhausted"):
                raise ValueError(
                    "server restart limit reached; inspect worker log and restart the region"
                )
            if run["phase"] == "validating":
                if state["state"] == "completed":
                    if not run["desired"]:
                        return
                    run["worker"].start(run["command"])
                    run["phase"] = "running"
                elif state["state"] == "failed":
                    raise ValueError("server input validation failed; see worker log")
                elif time.time() - run["started_at"] > 120:
                    raise ValueError("server input validation timed out")
                return
            if run["operation"] != "live":
                if state["state"] in {"completed", "failed"}:
                    with self.lock:
                        if (
                            run["operation"] == "calibration"
                            and state["state"] == "completed"
                        ):
                            edge_id = run["options"]["edge_id"]
                            latest = (
                                self.registry.snapshot()["edges"][edge_id].get(
                                    "last_calibration"
                                )
                                or {}
                            )
                            if (
                                latest.get("completed_at", 0) < run["started_at"]
                                or latest.get("edge_sync", {}).get("status")
                                != "applied"
                            ):
                                raise ValueError(
                                    "calibration process ended without an applied edge result"
                                )
                            # Validate the merge before accepting the result for future runs.
                            edge = next(
                                e
                                for e in run["region"]["edges"]
                                if e["edge_id"] == edge_id
                            )
                            merge_edge_calibration(
                                self._calibration(run["region"]),
                                read_json(latest["result_path"]),
                                edge_id,
                                [c["camera_id"] for c in edge["cameras"]],
                            )
                            self.calibration_overrides.setdefault(run["region_id"], {})[
                                edge_id
                            ] = latest["result_path"]
                            atomic_json(
                                self.calibration_path, self.calibration_overrides
                            )
                        run.update(
                            desired=False, state=state["state"], error=state["error"]
                        )
                        self._persist(run)
                return
            registry = self.registry.snapshot()["edges"]
            for edge in run["region"]["edges"]:
                if not run["desired"]:
                    return
                edge_id = edge["edge_id"]
                record = registry.get(edge_id, {})
                if not record.get("approved") or not record.get("online"):
                    run["edge_errors"][edge_id] = "edge is offline or awaiting approval"
                    continue
                if time.monotonic() - run["last_commands"].get(edge_id, -100) < 5:
                    continue
                run["last_commands"][edge_id] = time.monotonic()
                try:
                    result = self.transport.request(
                        edge_id,
                        "ensure_inference",
                        {
                            "run_id": run["run_id"],
                            "region_id": run["region_id"],
                            "deployment_sha256": run["deployment_sha256"],
                            "calibration": run["calibration"],
                            "camera_ids": [
                                int(c["camera_id"]) for c in edge["cameras"]
                            ],
                        },
                    )
                    run["edge_results"][edge_id] = result
                    if result.get("error"):
                        run["edge_errors"][edge_id] = result["error"]
                    else:
                        run["edge_errors"].pop(edge_id, None)
                except Exception as exc:
                    run["edge_errors"][edge_id] = str(exc)
        except Exception as exc:
            run["worker"].stop()
            if run["recorder"]:
                run["recorder"].close()
            with self.lock:
                run.update(desired=False, state="failed", error=str(exc))
                self._persist(run)
            # No more lease renewals: edges stop even when commands cannot reach them.

    def status(self, region_id):
        with self.lock:
            region = self.catalog.get(region_id)
            run = self.runs.get(region_id)
            records = self.registry.snapshot()["edges"]
            received, inputs = self.scenes.get(region["scene_topic"], (None, {}))
            age = None if received is None else time.monotonic() - received
            fresh = age is not None and age < 3
            fresh = bool(
                fresh and run and run["desired"] and run["operation"] == "live"
            )
            edges = []
            for edge in region["edges"]:
                record = records.get(edge["edge_id"], {})
                runtime = (record.get("last_status") or {}).get("inference", {})
                if run and edge["edge_id"] in run["edge_results"]:
                    runtime = run["edge_results"][edge["edge_id"]]
                observed = record.get("cameras") or {}
                edges.append(
                    {
                        **edge,
                        "approved": bool(record.get("approved")),
                        "online": bool(record.get("online")),
                        "inference": runtime,
                        "data_fresh": bool(
                            fresh and inputs.get(edge["edge_id"]) == "online"
                        ),
                        "error": run["edge_errors"].get(edge["edge_id"])
                        if run
                        else None,
                        "cameras": [
                            {**camera, "status": observed.get(camera["camera_id"], {})}
                            for camera in edge["cameras"]
                        ],
                    }
                )
            state = run["state"] if run else "stopped"
            worker = run["worker"].status() if run else None
            if run and run["desired"] and run["operation"] == "live":
                healthy = sum(
                    e["data_fresh"]
                    and e["online"]
                    and e["approved"]
                    and not e["error"]
                    and e["inference"].get("run_id") == run["run_id"]
                    for e in edges
                )
                if worker["state"] == "failed":
                    state = "failed"
                elif (
                    run["phase"] == "running"
                    and worker["state"] == "running"
                    and healthy == len(edges)
                ):
                    state = "running"
                elif healthy or time.time() - run["started_at"] > 180:
                    state = "degraded"
                else:
                    state = "starting"
            elif run and run["desired"]:
                state = worker["state"]
            return {
                **region,
                "edges": edges,
                "state": state,
                "worker": worker,
                "run_id": run["run_id"] if run else None,
                "operation": run["operation"] if run else None,
                "recording": bool(
                    run
                    and run["desired"]
                    and run["recorder"]
                    and run["recorder"].status()["active"]
                ),
                "recording_status": run["recorder"].status()
                if run and run["recorder"]
                else None,
                "last_scene_age_seconds": age,
                "error": run["error"] if run else None,
            }

    def regions(self):
        return [self.status(r["region_id"]) for r in self.catalog.all()]

    def recordings(self, region_id):
        self.catalog.get(region_id)
        result = []
        for path in self.data_dir.glob("runs/*/manifest.json"):
            record = read_json(path)
            scene = path.parent / "scenes.jsonl"
            if record["region_id"] == region_id:
                current = self.runs.get(region_id)
                if current and current["run_id"] == record["run_id"]:
                    record["status"] = self.status(region_id)["state"]
                result.append(
                    {
                        **record,
                        "scene_bytes": scene.stat().st_size if scene.exists() else 0,
                    }
                )
        return sorted(result, key=lambda r: r["started_at"], reverse=True)

    def recording_path(self, region_id, run_id):
        validate_edge_id(run_id)
        directory = self.data_dir / "runs" / run_id
        metadata = read_json(directory / "manifest.json")
        if metadata["region_id"] != region_id:
            raise KeyError("recording does not belong to region")
        path = directory / "scenes.jsonl"
        if not path.is_file():
            raise ValueError("run has no recorded scenes")
        return path

    def replay(self, region_id, recording_id):
        with self.lock:
            source = self.recording_path(region_id, recording_id)
            current = self.runs.get(region_id)
            if current and current["run_id"] == recording_id and current["desired"]:
                raise ValueError("stop recording before replay")
            run, created = self._new_run(
                region_id, "replay", {"recording_id": recording_id}
            )
            if created:
                command = [
                    sys.executable,
                    "-m",
                    "apps.edge_manager.app.scene_replay",
                    "--input",
                    str(source),
                    "--endpoint",
                    self.endpoint,
                    "--topic",
                    run["region"]["scene_topic"],
                ]
                self._start_job(run, command)
            return self.status(region_id)

    def _start_job(self, run, command):
        try:
            run["worker"].start(command, restart=False)
            run["phase"] = "running"
        except Exception as exc:
            run.update(desired=False, state="failed", error=str(exc))
            self._persist(run)
            raise

    def calibrate(self, region_id, edge_id, reference_video, marker_tree, checkpoint):
        with self.lock:
            region = self.catalog.get(region_id)
            if edge_id not in {e["edge_id"] for e in region["edges"]}:
                raise ValueError("edge does not belong to region")
            edge = self.registry.snapshot()["edges"].get(edge_id, {})
            if not edge.get("approved") or not edge.get("online"):
                raise ValueError("calibration requires an approved, online edge")
            paths = {
                k: str(Path(v).expanduser().resolve())
                for k, v in {
                    "reference_video": reference_video,
                    "marker_tree": marker_tree,
                    "checkpoint": checkpoint,
                }.items()
            }
            for value in paths.values():
                if not Path(value).is_file():
                    raise ValueError(f"calibration input does not exist: {value}")
            run, created = self._new_run(
                region_id, "calibration", {"edge_id": edge_id, **paths}
            )
            if created:
                command = [
                    os.getenv("DT_CALIBRATION_PYTHON", sys.executable),
                    str(PROJECT_ROOT / "apps/calibration_worker/inference.py"),
                    "--mode",
                    "online",
                    "--edge-id",
                    edge_id,
                    "--registry",
                    str(self.registry.path),
                    "--endpoint",
                    self.endpoint,
                    "--topic-root",
                    region["topic_root"],
                    "--output-dir",
                    str(run["directory"] / "calibration"),
                ]
                for key, value in paths.items():
                    command.extend(["--" + key.replace("_", "-"), value])
                self._start_job(run, command)
            return self.status(region_id)

    def logs(self, region_id, edge_id=None):
        with self.lock:
            region = self.catalog.get(region_id)
            if edge_id:
                if edge_id not in {e["edge_id"] for e in region["edges"]}:
                    raise ValueError("edge does not belong to region")
            else:
                run = self.runs.get(region_id)
                return {"text": run["worker"].logs() if run else ""}
        return self.transport.request(edge_id, "inference_logs")

    def configure_region(self, region_id, settings=None, *, remove=False):
        with self.lock:
            run = self.runs.get(region_id)
            if run and (run["desired"] or run["state"] == "stopping"):
                raise ValueError("stop the region before changing its configuration")
            if remove:
                self.catalog.remove(region_id)
                self.runs.pop(region_id, None)
                return {"removed": region_id}
            expanded = self.catalog._expand(region_id, settings)
            if expanded["topic_root"] != self.transport.topic_root:
                raise ValueError("deployment topic root must match the manager")
            previous = self.catalog.expanded.get(region_id)
            result = self.catalog.put(region_id, settings)
            if previous and previous["deployment"] != result["deployment"]:
                self.calibration_overrides.pop(region_id, None)
                atomic_json(self.calibration_path, self.calibration_overrides)
            return result

    def edge_action(self, edge_id, action, *, name=None, endpoint=None):
        validate_edge_id(edge_id)
        record = self.registry.snapshot()["edges"][edge_id]
        if action in {"approve", "revoke"}:
            status = record.get("last_status") or {}
            address = endpoint or status.get("zenoh_endpoint")
            if not address:
                raise ValueError("edge endpoint is unknown")
            if action == "revoke":
                # Withhold leases immediately, even if the remote ACK cannot arrive.
                self.registry.update(edge_id, approved=False)
            self.transport.request(
                edge_id,
                action,
                {
                    "approved": action == "approve",
                    "display_name": name or record["display_name"],
                    "zenoh_endpoint": address,
                    "topic_root": self.transport.topic_root,
                },
                configuration=True,
            )
            return self.registry.update(
                edge_id,
                approved=action == "approve",
                display_name=name or record["display_name"],
            )
        if action == "remove":
            if any(
                edge_id == e["edge_id"] for r in self.catalog.all() for e in r["edges"]
            ):
                raise ValueError(
                    "remove the edge from its deployment before removing registration"
                )
            self.registry.remove(edge_id)
            return {"removed": edge_id}
        if action not in {"ping", "info", "shutdown"}:
            raise ValueError("unsupported edge action")
        if not record.get("approved"):
            raise ValueError("edge is not approved")
        return self.transport.request(edge_id, action)

    def close(self):
        with self.lock:
            self.closing = True
            for run in self.runs.values():
                if run["desired"]:
                    run.update(desired=False, state="stopping")
        self.pool.shutdown(wait=True)
        for run in self.runs.values():
            if run["state"] == "stopping":
                self._reconcile(run)
