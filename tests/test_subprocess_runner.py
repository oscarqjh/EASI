"""Tests for SubprocessRunner env var injection."""

from __future__ import annotations

import os
from pathlib import Path

from easi.core.render_platforms import EnvVars, get_render_platform
from easi.simulators.subprocess_runner import SubprocessRunner


class TestSubprocessRunnerEnvVars:
    """Tests for env var injection into bridge subprocess."""

    def test_constructor_accepts_env_vars(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            extra_env=EnvVars(replace={"MY_VAR": "my_value"}),
        )
        assert runner.extra_env.replace == {"MY_VAR": "my_value"}

    def test_default_env_is_none(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
        )
        assert runner.extra_env is None

    def test_build_env_merges_with_os_environ(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            extra_env=EnvVars(replace={"COPPELIASIM_ROOT": "/opt/coppeliasim"}),
        )
        env = runner._build_subprocess_env()
        assert env["COPPELIASIM_ROOT"] == "/opt/coppeliasim"
        assert "PATH" in env

    def test_build_env_returns_none_when_no_extra(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
        )
        assert runner._build_subprocess_env() is None

    def test_extra_env_prepends_to_path_vars(self):
        runner = SubprocessRunner(
            python_executable="/usr/bin/python3",
            bridge_script_path=Path("/dev/null"),
            render_platform=get_render_platform("headless"),
            extra_env=EnvVars(prepend={"LD_LIBRARY_PATH": "/opt/sim/lib"}),
        )
        env = runner._build_subprocess_env()
        ld_path = env.get("LD_LIBRARY_PATH", "")
        assert ld_path.startswith("/opt/sim/lib")

    def test_non_path_var_replaces(self):
        """Non-path env vars should replace, not prepend."""
        os.environ["MY_EXISTING"] = "old_value"
        try:
            runner = SubprocessRunner(
                python_executable="/usr/bin/python3",
                bridge_script_path=Path("/dev/null"),
                render_platform=get_render_platform("headless"),
                extra_env=EnvVars(replace={"MY_EXISTING": "new_value"}),
            )
            env = runner._build_subprocess_env()
            assert env["MY_EXISTING"] == "new_value"
        finally:
            del os.environ["MY_EXISTING"]
