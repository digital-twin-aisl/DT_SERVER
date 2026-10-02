# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Lease-bound inference owned by the persistent edge agent."""

import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from dt_common.infrastructure.process import ManagedProcess
from dt_common.infrastructure.deployment import materialize_deployment
from apps.edge_client.infrastructure.identity import load_edge_metadata
from dt_common.contracts.edge import validate_edge_id

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EdgeInferenceRuntime:
    def __init__(
        self,
        edge_id,
        identity_path,
        camera_config,
        endpoint,
        *,
        process_factory=ManagedProcess,
    ):
        self.edge_id = edge_id
        self.identity = Path(identity_path).resolve()
        self.camera_config = str(Path(camera_config).resolve())
        self.endpoint = endpoint
        self.directory = self.identity.parent / "runtime" / edge_id
        self.worker = process_factory(
            self.directory / "inference.log", cwd=PROJECT_ROOT
        )
        self.lock = threading.RLock()
        self.run_id = self.region_id = None
        self.deadline = 0.0
        self.stage = "stopped"
        self.error = None
        self.retired = []
        self.command = None
        self.validation_started = 0.0

    def ensure(self, parameters):
        with self.lock:
            run_id = validate_edge_id(str(parameters["run_id"]))
            region_id = validate_edge_id(str(parameters["region_id"]))
            if run_id in self.retired:
                raise ValueError("run has already stopped; start a new run")
            if self.run_id:
                if (run_id, region_id) != (self.run_id, self.region_id):
                    raise ValueError("edge is owned by another active run")
                self.deadline = time.monotonic() + 30
                return self.status()
            metadata = load_edge_metadata(self.identity)
            if not metadata.get("approved"):
                raise PermissionError("edge is not approved")
            settings = metadata.get("inference") or {}
            deployment = (self.identity.parent / settings["deployment"]).resolve()
            if (
                hashlib.sha256(deployment.read_bytes()).hexdigest()
                != parameters["deployment_sha256"]
            ):
                raise ValueError(
                    "deployment differs from manager; sync the registered deployment before starting"
                )
            manifest = json.loads(deployment.read_text())
            assigned = next(
                e["camera_ids"]
                for e in manifest["edges"]
                if e["id"] == self.edge_id and e.get("enabled", True)
            )
            if assigned != parameters["camera_ids"]:
                raise ValueError("camera assignment differs from manager")
            if manifest.get("topic_root", "dt/edges") != metadata.get(
                "topic_root", "dt/edges"
            ):
                raise ValueError("agent topic root differs from deployment")
            # The manager supplies one immutable calibration snapshot to every
            # participant, using this host's original map/cache paths.
            if parameters.get("calibration") is not None:
                deployment = materialize_deployment(
                    deployment,
                    self.directory / run_id / "spatial",
                    parameters["calibration"],
                )
            arguments = {
                "deployment": str(deployment),
                "edge_id": self.edge_id,
                "edge_id_file": str(self.identity),
                "camera_config": self.camera_config,
                "require_identity": True,
                "zenoh_endpoint": self.endpoint,
                "rtsp_max_frame_age": 0.75,
                "rtsp_max_skew": 0.03,
                "rtsp_buffer_frames": 4,
            }
            if not self.endpoint:
                raise ValueError("agent Zenoh endpoint is not configured")
            if "tensorrt" in settings:
                arguments["tensorrt"] = settings["tensorrt"]
            if settings.get("cfg_focus"):
                arguments["cfg_focus"] = str(
                    (self.identity.parent / settings["cfg_focus"]).resolve()
                )
            self.directory.mkdir(parents=True, exist_ok=True)
            profile = self.directory / "managed.json"
            profile.write_text(
                json.dumps({"schema_version": 1, "arguments": arguments})
            )
            self.command = [
                sys.executable,
                str(PROJECT_ROOT / "apps/edge_client/inference.py"),
                "--runtime-config",
                str(profile),
            ]
            self.worker.start(self.command + ["--validate-only"], restart=False)
            self.run_id, self.region_id = run_id, region_id
            self.deadline = time.monotonic() + 30
            self.validation_started = time.monotonic()
            self.stage, self.error = "validating", None
            return self.status()

    def tick(self):
        with self.lock:
            if not self.run_id:
                return
            if time.monotonic() >= self.deadline:
                self.stop(retire=False)
                self.error = "manager lease expired; inference stopped"
                return
            self.worker.tick()
            state = self.worker.status()
            if self.stage == "validating":
                if state["state"] == "completed":
                    self.worker.start(self.command)
                    self.stage = "running"
                elif state["state"] == "failed":
                    self.stage, self.error = (
                        "failed",
                        "input validation failed; inspect inference log",
                    )
                elif time.monotonic() - self.validation_started > 120:
                    self.worker.stop()
                    self.stage, self.error = "failed", "input validation timed out"

    def stop(self, run_id=None, *, retire=True):
        with self.lock:
            if run_id and self.run_id and run_id != self.run_id:
                raise ValueError("stop command belongs to another run")
            if run_id and retire:
                self.retired = (self.retired + [run_id])[-128:]
            elif self.run_id and retire:
                self.retired = (self.retired + [self.run_id])[-128:]
            self.worker.stop()
            self.run_id = self.region_id = None
            self.stage = "stopped"
            return self.status()

    def status(self):
        with self.lock:
            state = self.worker.status()
            return {
                **state,
                "stage": self.stage,
                "run_id": self.run_id,
                "region_id": self.region_id,
                "error": self.error or state["error"],
                "lease_remaining_seconds": max(0, self.deadline - time.monotonic())
                if self.run_id
                else 0,
            }
