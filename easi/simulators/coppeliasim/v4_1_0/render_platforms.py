"""Custom render platforms for CoppeliaSim V4.1.0.

CoppeliaSim needs simulator-specific env vars (QT_QPA_PLATFORM_PLUGIN_PATH,
__EGL_VENDOR_LIBRARY_FILENAMES) that depend on the CoppeliaSim binary location.
These custom platforms compute the correct paths from the env_manager.
"""

from __future__ import annotations

import os
from pathlib import Path

from easi.core.render_platform import (
    AutoPlatform,
    EnvVars,
    NativePlatform,
    XvfbPlatform,
)


def _coppeliasim_qt_env_vars(env_manager, include_mesa_egl: bool = False) -> EnvVars:
    """Compute CoppeliaSim Qt env vars.

    CoppeliaSim always needs QT_QPA_PLATFORM_PLUGIN_PATH to find its bundled
    Qt plugins (libqxcb.so, etc.), regardless of display mode.

    Args:
        env_manager: The CoppeliaSimEnvManager instance.
        include_mesa_egl: If True, also set __EGL_VENDOR_LIBRARY_FILENAMES
            to Mesa vendor (needed for Xvfb, NVIDIA EGL crashes).

    Returns:
        EnvVars with QT_QPA_PLATFORM_PLUGIN_PATH and optionally Mesa EGL.
    """
    if env_manager is None:
        return EnvVars()
    binary_dir_name = env_manager.installation_kwargs.get("binary_dir_name", "")
    if not binary_dir_name:
        return EnvVars()
    t = env_manager._get_template_variables()
    coppeliasim_root = env_manager._resolve_template(
        "{extras_dir}/" + binary_dir_name, t
    )
    replace: dict[str, str] = {}
    if include_mesa_egl:
        mesa_vendor = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        if mesa_vendor.exists():
            replace["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa_vendor)
    return EnvVars(
        replace=replace,
        prepend={"QT_QPA_PLATFORM_PLUGIN_PATH": coppeliasim_root},
    )


class CoppeliaSimNativePlatform(NativePlatform):
    """Native display for CoppeliaSim — Qt plugin path but no Mesa override."""

    @property
    def name(self) -> str:
        return "native"

    def get_env_vars(self) -> EnvVars:
        # CoppeliaSim always needs Qt plugin path, even with native display
        return _coppeliasim_qt_env_vars(self._env_manager, include_mesa_egl=False)


class CoppeliaSimXvfbPlatform(XvfbPlatform):
    """Xvfb platform for CoppeliaSim — sets Qt plugin path + Mesa EGL vendor."""

    @property
    def name(self) -> str:
        return "xvfb"

    def get_env_vars(self) -> EnvVars:
        return _coppeliasim_qt_env_vars(self._env_manager, include_mesa_egl=True)


class CoppeliaSimAutoPlatform(AutoPlatform):
    """Auto-detect for CoppeliaSim — native if DISPLAY, xvfb otherwise.

    Both modes need Qt plugin path; only xvfb needs Mesa EGL override.
    """

    @property
    def name(self) -> str:
        return "auto"

    def get_env_vars(self) -> EnvVars:
        include_mesa = not os.environ.get("DISPLAY", "")
        return _coppeliasim_qt_env_vars(self._env_manager, include_mesa_egl=include_mesa)
