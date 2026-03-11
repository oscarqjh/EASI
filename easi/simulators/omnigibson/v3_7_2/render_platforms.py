"""Custom render platforms for OmniGibson v3.7.2.

Each platform sets OMNIGIBSON_HEADLESS so Isaac Sim starts in the correct mode:
- native: OMNIGIBSON_HEADLESS=0 (user has a real display, Isaac Sim GUI opens)
- auto:   OMNIGIBSON_HEADLESS=0 if DISPLAY is set, 1 otherwise
"""

from __future__ import annotations

import os

from easi.core.render_platforms import AutoPlatform, EnvVars, NativePlatform


class OmniGibsonNativePlatform(NativePlatform):
    """Native display for OmniGibson — Isaac Sim GUI window, OMNIGIBSON_HEADLESS=0."""

    @property
    def name(self) -> str:
        return "native"

    def get_env_vars(self) -> EnvVars:
        return EnvVars(replace={"OMNIGIBSON_HEADLESS": "0"})


class OmniGibsonAutoPlatform(AutoPlatform):
    """Auto-detect for OmniGibson — native mode if $DISPLAY is set, headless otherwise."""

    @property
    def name(self) -> str:
        return "auto"

    def get_env_vars(self) -> EnvVars:
        has_display = bool(os.environ.get("DISPLAY", ""))
        return EnvVars(replace={"OMNIGIBSON_HEADLESS": "0" if has_display else "1"})
