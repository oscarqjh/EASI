"""Environment manager for AI2-THOR v5.0.0 (modern API)."""

from __future__ import annotations

from pathlib import Path

from easi.core.base_env_manager import BaseEnvironmentManager


class AI2ThorEnvManagerV500(BaseEnvironmentManager):
    """Environment manager for AI2-THOR 5.0.0."""

    @property
    def simulator_name(self) -> str:
        return "ai2thor"

    @property
    def version(self) -> str:
        return "v5_0_0"

    @property
    def needs_display(self) -> bool:
        return True  # AI2-THOR Unity requires X11

    @property
    def xvfb_screen_config(self) -> str:
        return "1280x720x24"  # v5 supports higher resolution

    def get_conda_env_yaml_path(self) -> Path:
        return Path(__file__).parent / "conda_env.yaml"

    def get_requirements_txt_path(self) -> Path:
        return Path(__file__).parent / "requirements.txt"

    def get_system_deps(self) -> list[str]:
        return ["conda", "xvfb"]

    def get_validation_import(self) -> str:
        return "import ai2thor; assert ai2thor.__version__.startswith('5.')"
