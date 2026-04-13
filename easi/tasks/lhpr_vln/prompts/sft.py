"""SFT-aligned prompt builder for LHPR-VLN.

Replicates the exact prompt structure used during SFT training
(vln_si_sft/prepare_jsonl_data_parallel.py) so that fine-tuned
checkpoints see the same format at inference time.

Key differences from the default prompt builder:
- Uses special action tokens (<|forward|>, <|left|>, <|right|>, <|stop|>)
- Includes historical front-view images (sampled, up to max_history_images)
- Predicts window_size actions per LLM call (multi-action)
- Output format: <action>..tokens..</action> (no JSON, no reasoning)
- History resets on stop (matching training behavior)
- No environmental feedback (geodesic distance, etc.)
"""
from __future__ import annotations

import re

from easi.agents.prompt_builder import validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)

# Action name <-> special token mapping
_ACTION_TO_TOKEN = {
    "move_forward": "<|forward|>",
    "turn_left": "<|left|>",
    "turn_right": "<|right|>",
    "stop": "<|stop|>",
}
_TOKEN_TO_ACTION = {v: k for k, v in _ACTION_TO_TOKEN.items()}

# Regex to extract action tokens from <action>...</action>
_ACTION_BLOCK_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_TOKEN_RE = re.compile(r"<\|(\w+)\|>")


def _sample_evenly(frames: list[str], count: int) -> list[str]:
    """Sample frames evenly from the list. Matches training implementation."""
    if count >= len(frames):
        return frames
    indices = [int(i * len(frames) / count) for i in range(count)]
    return [frames[i] for i in indices]


def _encode_image_base64(path: str) -> str | None:
    """Encode image file to base64 data URL."""
    from easi.agents.prompt_builder import _encode_image_base64
    return _encode_image_base64(path)


class LHPRVLNSFTPromptBuilder:
    """Prompt builder aligned with VLN SFT training format.

    Produces prompts identical to those used during supervised fine-tuning,
    with historical image context, special action tokens, and multi-action
    window prediction.
    """

    def __init__(
        self,
        window_size: int = 5,
        max_history_images: int = 20,
        **kwargs,
    ):
        self.window_size = window_size
        self.max_history_images = max_history_images

        # Track state across steps within an episode
        self._history_images: list[str] = []  # front-view paths from past steps
        self._stop_count: int = 0
        self._actions: list[str] = []

    def set_action_space(self, actions: list[str]) -> None:
        self._actions = list(actions)

    # ---- PromptBuilderProtocol methods ----

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        if self._actions != memory.action_space:
            self.set_action_space(memory.action_space)

        # Update history from memory steps
        self._update_history(memory)

        # Sample historical images
        history_paths = self._sample_history()

        # Collect current view paths
        current_views: list[str] = []
        if memory.current_observation and memory.current_observation.metadata:
            meta = memory.current_observation.metadata
            for key in ("left_rgb_path", "front_rgb_path", "right_rgb_path"):
                path = meta.get(key)
                if path:
                    current_views.append(path)

        # Build interleaved content blocks matching InternVL convention.
        # The chat template converts each image_url block to <image>\n,
        # so we place them at the exact positions where training had
        # literal <image> tokens.
        content: list[dict] = self._build_content(
            memory.task_description, history_paths, current_views,
        )

        return [{"role": "user", "content": content}]

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse <action>..tokens..</action> response into validated Actions."""
        # Extract action block
        match = _ACTION_BLOCK_RE.search(llm_response)
        if not match:
            # Try parsing the raw response as tokens (no wrapper tags)
            token_str = llm_response.strip()
        else:
            token_str = match.group(1)

        # Extract individual tokens
        tokens = _TOKEN_RE.findall(token_str)
        if not tokens:
            logger.warning("No action tokens found in response: %s", llm_response[:200])
            return []

        actions = []
        for token_name in tokens:
            full_token = f"<|{token_name}|>"
            action_name = _TOKEN_TO_ACTION.get(full_token)
            if action_name is None:
                logger.warning("Unknown action token: %s", full_token)
                continue
            validated = validate_action_name(action_name, memory.action_space)
            if validated:
                actions.append(Action(action_name=validated))
            else:
                logger.warning("Action '%s' not in action space", action_name)
                break

        return actions

    def get_response_format(self, memory: AgentMemory) -> dict | None:
        """No structured output — model outputs raw action tokens."""
        return None

    def get_reprompt_message(self) -> str:
        """Correction prompt for failed responses, matching the SFT format."""
        return (
            "Your previous response could not be parsed. "
            "You must output action tokens wrapped in <action> tags. "
            "For example: <action><|forward|><|forward|><|left|><|forward|><|forward|></action> "
            "or <action><|stop|></action>"
        )

    # ---- Internal methods ----

    def _build_content(
        self,
        instruction: str,
        history_paths: list[str],
        current_views: list[str],
    ) -> list[dict]:
        """Build interleaved text + image content blocks.

        Places image_url blocks at the exact positions where training had
        <image> tokens. The InternVL chat template converts each image_url
        block to <image>\\n, producing the same token sequence as training.
        No <image> tokens appear in the text itself.
        """
        content: list[dict] = []

        # Preamble
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

        # Historical images (interleaved)
        if history_paths:
            content.append({"type": "text", "text": "\n# Your historical pictures are: "})
            for path in history_paths:
                img_url = _encode_image_base64(path)
                if img_url:
                    content.append({"type": "image_url", "image_url": {"url": img_url}})

        # Current 3 views (interleaved)
        view_labels = ["left side", "front side", "right side"]
        content.append({"type": "text", "text": "\n# Your current observations are "})
        for i, path in enumerate(current_views):
            if i > 0:
                content.append({"type": "text", "text": ", "})
            content.append({"type": "text", "text": f"{view_labels[i]}: "})
            img_url = _encode_image_base64(path)
            if img_url:
                content.append({"type": "image_url", "image_url": {"url": img_url}})

        # Mission + closing
        content.append({"type": "text", "text": (
            f"\n# Your mission is: {instruction}\n"
            "PS: The mission is complex. You may infer several sub-tasks "
            "within the mission, and output <|stop|> when a sub-task is "
            f"achieved. So far, you have output <|stop|> {self._stop_count} "
            "times. Historical information reflects progress up to the "
            "current subgoal. <|NAV|>"
        )})

        return content

    def _update_history(self, memory: AgentMemory) -> None:
        """Update history images and stop count from memory steps.

        Rebuilds from scratch each call (stateless relative to memory).
        Matches training behavior: history is front-view images,
        resets on stop.
        """
        self._history_images = []
        self._stop_count = 0

        for step in memory.steps:
            if step.observation and step.observation.metadata:
                front_path = step.observation.metadata.get("front_rgb_path")
                if front_path:
                    self._history_images.append(front_path)

            # Reset history on stop (matching training: full_history = [])
            if step.action and step.action.action_name == "stop":
                self._stop_count += 1
                self._history_images = []

    def _sample_history(self) -> list[str]:
        """Sample historical images matching training strategy.

        If history exceeds max_history_images:
        - Keep most recent max_history_images//2 frames
        - Evenly sample the rest from older frames
        """
        if len(self._history_images) <= self.max_history_images:
            return list(self._history_images)

        recent_count = self.max_history_images // 2
        sampled_count = self.max_history_images - recent_count

        recent = self._history_images[-recent_count:]
        earlier = self._history_images[:-recent_count]
        sampled = _sample_evenly(earlier, sampled_count)

        return sampled + recent
