"""Subprocess lifecycle manager for bridge processes.

Manages the full lifecycle of a simulator bridge subprocess:
- Launch: Spawns the bridge in a new process group (for clean shutdown)
- Health check: Polls status.json with configurable timeout
- Send command: Writes command.json, polls response.json
- Crash recovery: Detects subprocess exit during polling
- Cleanup: SIGTERM → wait → SIGKILL the entire process group; removes temp workspace

Supports xvfb-run wrapping for simulators that need a display.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from pathlib import Path

from easi.communication.filesystem import (
    cleanup_workspace,
    create_workspace,
    poll_for_response,
    poll_for_status,
    write_command,
)
from easi.core.exceptions import SimulatorError, SimulatorTimeoutError

logger = logging.getLogger("easi.simulators.subprocess_runner")


class SubprocessRunner:
    """Manages a bridge subprocess for a single simulator instance."""

    def __init__(
        self,
        python_executable: str,
        bridge_script_path: Path,
        needs_display: bool = False,
        xvfb_screen_config: str = "1024x768x24",
        startup_timeout: float = 30.0,
        command_timeout: float = 60.0,
        poll_interval: float = 0.1,
    ):
        self.python_executable = python_executable
        self.bridge_script_path = bridge_script_path
        self.needs_display = needs_display
        self.xvfb_screen_config = xvfb_screen_config
        self.startup_timeout = startup_timeout
        self.command_timeout = command_timeout
        self.poll_interval = poll_interval

        self._process: subprocess.Popen | None = None
        self._workspace: Path | None = None

    @property
    def workspace(self) -> Path | None:
        """The IPC workspace directory for this runner."""
        return self._workspace

    def launch(self) -> None:
        """Launch the bridge subprocess and wait for it to report ready."""
        if self._process is not None:
            raise RuntimeError("Subprocess already running")

        self._workspace = create_workspace()
        cmd = self._build_launch_command()

        logger.info(
            "Launching bridge: %s (workspace: %s)",
            self.bridge_script_path.name,
            self._workspace,
        )
        logger.debug("Full command: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,  # new session = new process group
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the bridge to signal readiness
        try:
            status = poll_for_status(
                self._workspace,
                poll_interval=self.poll_interval,
                timeout=self.startup_timeout,
                process=self._process,
            )
            if not status.get("ready", False):
                stderr = self._read_stderr()
                raise SimulatorError(
                    f"Bridge reported not ready. stderr:\n{stderr}"
                )
            logger.info("Bridge subprocess ready (PID: %d)", self._process.pid)
        except (SimulatorError, SimulatorTimeoutError):
            self.shutdown()
            raise

    def send_command(self, command: dict, timeout: float | None = None) -> dict:
        """Send a command to the bridge and wait for the response.

        Args:
            command: The command dict to send.
            timeout: Override the default command timeout.

        Returns:
            The response dict from the bridge.
        """
        if self._process is None or self._workspace is None:
            raise RuntimeError("Subprocess not running")

        write_command(self._workspace, command)

        return poll_for_response(
            self._workspace,
            poll_interval=self.poll_interval,
            timeout=timeout or self.command_timeout,
            process=self._process,
        )

    def is_alive(self) -> bool:
        """Check if the bridge subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def shutdown(self) -> None:
        """Shut down the bridge subprocess and clean up.

        Kills the entire process group (bridge + any child processes like
        Unity binaries) to prevent zombie processes.
        """
        self._terminate_process_tree()
        self._process = None

        if self._workspace is not None:
            cleanup_workspace(self._workspace)
            self._workspace = None

    def _build_launch_command(self) -> list[str]:
        """Build the command to launch the bridge subprocess."""
        cmd = [
            self.python_executable,
            str(self.bridge_script_path),
            "--workspace",
            str(self._workspace),
        ]

        if self.needs_display and not self._has_display():
            # Wrap with xvfb-run for headless environments
            cmd = [
                "xvfb-run", "-a",
                "-s", f"-screen 0 {self.xvfb_screen_config}",
            ] + cmd

        return cmd

    def _has_display(self) -> bool:
        """Check if a real display is available."""
        display = os.environ.get("DISPLAY", "")
        return bool(display) and not display.startswith(":")

    def _terminate_process_tree(self) -> None:
        """Kill the bridge and ALL its child processes."""
        if self._process is None:
            return

        try:
            pgid = os.getpgid(self._process.pid)
        except ProcessLookupError:
            return  # already dead

        try:
            # SIGTERM the entire group first (graceful)
            os.killpg(pgid, signal.SIGTERM)
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # SIGKILL the entire group (force)
            try:
                os.killpg(pgid, signal.SIGKILL)
                self._process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        except ProcessLookupError:
            pass  # already dead

        logger.debug("Process tree terminated")

    def _read_stderr(self) -> str:
        """Read stderr from the subprocess (non-blocking)."""
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            return "<could not read stderr>"
