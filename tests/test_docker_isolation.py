# tests/test_docker_isolation.py
"""Tests for Docker simulator isolation layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSimulatorEntryRuntime:
    """SimulatorEntry supports runtime field."""

    def test_default_runtime_is_conda(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
        )
        assert entry.runtime == "conda"

    def test_runtime_can_be_docker(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
            runtime="docker",
        )
        assert entry.runtime == "docker"

    def test_data_dir_default_empty(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
        )
        assert entry.data_dir == ""

    def test_data_dir_can_be_set(self):
        from easi.simulators.registry import SimulatorEntry

        entry = SimulatorEntry(
            name="test_sim",
            version="v1",
            description="test",
            simulator_class="easi.simulators.dummy.v1.simulator.DummySimulator",
            env_manager_class="easi.simulators.dummy.v1.env_manager.DummyEnvManager",
            python_version="3.10",
            data_dir="/datasets/test",
        )
        assert entry.data_dir == "/datasets/test"
