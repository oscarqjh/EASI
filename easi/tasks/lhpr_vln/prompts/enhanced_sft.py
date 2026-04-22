"""Training-distribution-faithful SFT prompt builder for LHPR-VLN.

Mirrors fantasy-vln's ``display_env`` preprocessing so a checkpoint trained
on those enhanced frames receives matching data at eval time:

1. reads the sensor frame from disk (RGB or RGBA, depending on whether the
   bridge ran with ``save_rgba: true``),
2. applies a configurable contrast boost (default 1.5, as in
   ``lhvln/habitat_base/visualization.py``),
3. resizes to a square resolution (default 366, matching fantasy-vln),
4. re-encodes as a PNG data URL for the LLM message.

Behaviour is otherwise identical to ``LHPRVLNSFTPromptBuilder``. See
``docs/superpowers/specs/2026-04-21-lhpr-vln-enhanced-sft-design.md`` for
the motivation + distribution-gap measurements.
"""
from __future__ import annotations

import base64
import io

from easi.agents.prompt_builder import _encode_image_base64 as _base_encode
from easi.core.memory import AgentMemory
from easi.tasks.lhpr_vln.prompts.sft import LHPRVLNSFTPromptBuilder
from easi.utils.logging import get_logger

logger = get_logger(__name__)


def _encode_image_enhanced(
    path: str,
    contrast: float = 1.5,
    resize_to: int = 366,
) -> str | None:
    """Encode ``path`` as a contrast-boosted + resized PNG data URL.

    Uses the same NFS-safe reader the baseline SFT builder uses (it
    handles truncated-on-NFS retries), then re-applies PIL transforms
    before emitting the data URL. Preserves the input mode (RGB or
    RGBA) through the whole chain; ``ImageEnhance.Contrast`` only
    scales the RGB channels and leaves alpha untouched.

    ``contrast == 1.0`` skips the enhancer. ``resize_to`` falsy (0 or
    None) skips the resize. This keeps no-op kwargs bit-identical to
    the raw ``_encode_image_base64`` output (modulo mime — still PNG).
    """
    from PIL import Image, ImageEnhance

    data_url = _base_encode(path)
    if data_url is None:
        return None
    _, b64 = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))

    if contrast and contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if resize_to:
        img = img.resize((int(resize_to), int(resize_to)))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


class LHPRVLNEnhancedSFTPromptBuilder(LHPRVLNSFTPromptBuilder):
    """SFT builder that re-applies fantasy-vln's ``display_env`` pipeline."""

    def __init__(
        self,
        window_size: int = 5,
        max_history_images: int = 20,
        enhance_contrast: float = 1.5,
        resize_to: int = 366,
        **kwargs,
    ):
        super().__init__(
            window_size=window_size,
            max_history_images=max_history_images,
            **kwargs,
        )
        self.enhance_contrast = enhance_contrast
        self.resize_to = resize_to

    def _encode(self, path: str) -> str | None:
        return _encode_image_enhanced(
            path, contrast=self.enhance_contrast, resize_to=self.resize_to,
        )

    def _build_content(
        self,
        instruction: str,
        history_paths: list[str],
        current_views: list[str],
    ) -> list[dict]:
        # Structural copy of ``LHPRVLNSFTPromptBuilder._build_content`` — only
        # the image-encoder helper differs. Any future change to the parent
        # layout should also be ported here; the structural-parity test in
        # ``tests/test_lhpr_vln_enhanced.py`` guards against silent drift.
        content: list[dict] = []

        content.append({"type": "text", "text": (
            "You are an autonomous navigation robot. You will get a task "
            "with historical pictures and current pictures you see.\n"
            f"Based on this information, you need to decide your next "
            f"{self.window_size} actions, which could involve "
            "<|left|>,<|right|>,<|forward|>. "
            "If you finish your mission, output <|stop|>. "
            "Here are some examples: "
            "<|left|><|forward|><|forward|><|stop|>, "
            "<|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>"
        )})

        if history_paths:
            content.append({"type": "text", "text": "\n# Your historical pictures are: "})
            for path in history_paths:
                img_url = self._encode(path)
                if img_url:
                    content.append({"type": "image_url", "image_url": {"url": img_url}})

        view_labels = ["left side", "front side", "right side"]
        content.append({"type": "text", "text": "\n# Your current observations are "})
        for i, path in enumerate(current_views):
            if i > 0:
                content.append({"type": "text", "text": ", "})
            content.append({"type": "text", "text": f"{view_labels[i]}: "})
            img_url = self._encode(path)
            if img_url:
                content.append({"type": "image_url", "image_url": {"url": img_url}})

        content.append({"type": "text", "text": (
            f"\n# Your mission is: {instruction}\n"
            "PS: The mission is complex. You may infer several sub-tasks "
            "within the mission, and output <|stop|> when a sub-task is "
            f"achieved. So far, you have output <|stop|> {self._stop_count} "
            "times. Historical information reflects progress up to the "
            "current subgoal. <|NAV|>"
        )})

        return content
