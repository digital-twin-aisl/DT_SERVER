# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Small process owner used by both the manager and edge agents.

Commands are constructed by application code, never accepted as shell strings.
One owner terminates the entire process group, including camera subprocesses.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import threading
import time


class ManagedProcess:
    def __init__(self, log_path, *, cwd, max_restarts=3, restart_delay=3.0):
        self.log_path = Path(log_path)
        self.cwd = str(cwd)
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self._lock = threading.RLock()
        self.process = None
        self.command = None
        self.desired = False
        self.restarts = 0
        self.next_restart = 0.0
        self.exit_code = None
        self.error = None
        self.started_at = None
        self.completed = False
        self._group_cleaned = True
        self.log_error = None
        self._log_reader = None

    def _spawn(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_reader:
            self._log_reader.join(timeout=1)
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._group_cleaned = False
        self.log_error = None
        self._log_reader = threading.Thread(
            target=self._drain_log, args=(self.process.stdout,), daemon=True
        )
        self._log_reader.start()
        self.started_at = time.time()
        self.exit_code = None
        self.error = None

    def _drain_log(self, pipe):
        # Keep draining even after a filesystem failure so logging cannot block
        # a worker's stdout pipe. Bound each log to two 10 MiB segments.
        with pipe:
            while chunk := pipe.read1(65536):
                if self.log_error:
                    continue
                try:
                    if (
                        self.log_path.exists()
                        and self.log_path.stat().st_size + len(chunk) > 10 * 1024**2
                    ):
                        self.log_path.replace(
                            self.log_path.with_suffix(".previous.log")
                        )
                    with self.log_path.open("ab") as stream:
                        stream.write(chunk)
                except OSError as exc:
                    self.log_error = str(exc)

    def start(self, command, *, restart=True):
        with self._lock:
            if self.desired:
                if list(command) != self.command:
                    raise ValueError(
                        "stop the current process before changing its configuration"
                    )
                return self.status()
            self.command = list(command)
            self.restart = restart
            self.desired, self.completed = True, False
            self.restarts, self.next_restart = 0, 0.0
            try:
                self._spawn()
            except OSError as exc:
                self.desired = False
                self.error = str(exc)
                raise
            return self.status()

    def tick(self):
        with self._lock:
            if not self.desired or self.process is None:
                return
            code = self.process.poll()
            if code is None:
                return
            self.exit_code = code
            if not self.restart:
                self.completed = code == 0
                self.desired = False
                self.error = (
                    None if self.completed else f"process exited with code {code}"
                )
                self._kill_group()
                return
            if self.restarts >= self.max_restarts:
                self.error = f"process exited with code {code}; restart limit reached"
                self._kill_group()
                return
            if not self.next_restart:
                self.next_restart = time.monotonic() + self.restart_delay * (
                    2**self.restarts
                )
                self.error = f"process exited with code {code}; waiting to restart"
            if time.monotonic() >= self.next_restart:
                self._kill_group()
                self.restarts += 1
                self.next_restart = 0.0
                try:
                    self._spawn()
                except OSError as exc:
                    self.error = str(exc)

    def _kill_group(self, sig=signal.SIGKILL):
        if self.process is not None and not self._group_cleaned:
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            if sig == signal.SIGKILL:
                self._group_cleaned = True

    def stop(self):
        with self._lock:
            self.desired = False
            if self.process is not None:
                self._kill_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._kill_group()
                    self.process.wait(timeout=5)
                # The leader can exit before its camera subprocesses.
                self._kill_group()
                self.exit_code = self.process.returncode
                self.process = None
            if self._log_reader:
                self._log_reader.join(timeout=1)
            self.error = None
            self.next_restart = 0.0
            return self.status()

    def status(self):
        with self._lock:
            alive = self.process is not None and self.process.poll() is None
            return {
                "state": (
                    "running"
                    if alive
                    else "completed"
                    if self.completed
                    else "failed"
                    if self.error
                    else "stopped"
                ),
                "pid": self.process.pid if alive else None,
                "desired": self.desired,
                "restarts": self.restarts,
                "restart_exhausted": bool(
                    self.desired and not alive and self.restarts >= self.max_restarts
                ),
                "exit_code": self.exit_code,
                "error": self.error,
                "started_at": self.started_at,
                "log_path": str(self.log_path),
                "log_error": self.log_error,
            }

    def logs(self, limit=16384):
        if not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as stream:
            stream.seek(max(0, self.log_path.stat().st_size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
