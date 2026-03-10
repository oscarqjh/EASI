"""Xorg render platform — GPU-accelerated X11 display managed by EASI.

Two classes:

* ``XorgPlatform`` — the top-level platform resolved from ``--render-platform xorg``.
  Owns the ``XorgManager`` lifecycle via ``setup()`` / ``teardown()`` and hands
  out per-worker instances via ``for_worker()``.
* ``_XorgWorkerPlatform`` — lightweight per-worker instance with a fixed
  display number and GPU ID, created by ``XorgPlatform.for_worker()``.
"""

from __future__ import annotations

from easi.core.render_platform import EnvVars, RenderPlatform
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class _XorgWorkerPlatform(RenderPlatform):
    """Per-worker Xorg platform with a dedicated display and GPU."""

    def __init__(self, display_num: int, gpu_id: int):
        super().__init__()
        self.display_num = display_num
        self.gpu_id = gpu_id

    @property
    def name(self) -> str:
        return "xorg"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def get_env_vars(self) -> EnvVars:
        return EnvVars(replace={
            "DISPLAY": f":{self.display_num}",
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
            "EASI_GPU_DISPLAY": "1",
        })

    def is_available(self) -> bool:
        return True


class XorgPlatform(RenderPlatform):
    """Render platform backed by auto-managed Xorg servers.

    Call ``setup(gpu_ids=...)`` to start Xorg servers, then
    ``for_worker(worker_id)`` to get per-worker instances.
    ``teardown()`` stops all servers.
    """

    def __init__(self, env_manager=None):
        super().__init__(env_manager=env_manager)
        self._xorg_mgr = None
        self._instances: list = []

    @property
    def name(self) -> str:
        return "xorg"

    def wrap_command(self, cmd: list[str], screen_config: str) -> list[str]:
        return cmd

    def get_env_vars(self) -> EnvVars:
        return EnvVars()

    def is_available(self) -> bool:
        return True

    def setup(self, gpu_ids: list[int] | None = None) -> None:
        """Start one Xorg server per GPU."""
        from easi.core.xorg_manager import XorgManager
        resolved_gpu_ids = gpu_ids or [0]
        self._xorg_mgr = XorgManager(gpu_ids=resolved_gpu_ids)
        self._instances = self._xorg_mgr.start()

    def teardown(self) -> None:
        """Stop all Xorg servers."""
        if self._xorg_mgr is not None:
            self._xorg_mgr.stop()
            self._xorg_mgr = None
            self._instances = []

    def for_worker(self, worker_id: int) -> _XorgWorkerPlatform:
        """Return a per-worker platform bound to a specific Xorg instance."""
        if not self._instances:
            raise RuntimeError(
                "XorgPlatform.setup() must be called before for_worker()"
            )
        inst = self._instances[worker_id % len(self._instances)]
        return _XorgWorkerPlatform(display_num=inst.display, gpu_id=inst.gpu_id)
