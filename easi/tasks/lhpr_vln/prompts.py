"""LHPR-VLN prompt builder for LLM-based navigation agents.

Builds prompts with:
- Panoramic RGB observation (left + front + right concatenated)
- Task instruction with subtask targets
- Current subtask progress (from observation metadata)
- Action history with distance feedback
- 4-action space: move_forward, turn_left, turn_right, stop
"""
from __future__ import annotations

import json

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory
from easi.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a navigation agent in a 3D indoor environment. You observe the environment through three camera views (left at -60°, front at 0°, right at +60°) and must navigate to sequential target objects.

Your actions:
- move_forward: Move forward 0.25 meters
- turn_left: Turn left 30 degrees
- turn_right: Turn right 30 degrees
- stop: Declare current subtask complete (use when you believe you are within 1 meter of the current target)

Strategy:
- Navigate to each target object in order. After reaching one target, use "stop" to advance to the next.
- Use visual cues to identify rooms and objects.
- If you are far from the target, explore by turning to survey the environment, then move toward likely locations.
- Use "stop" only when you are confident you are close to the current target.

Respond with ONLY a JSON object: {"action": "<action_name>"}"""


class LHPRVLNPromptBuilder:
    """Prompt builder for LHPR-VLN navigation tasks."""

    def __init__(self, use_feedback: bool = True, chat_history: bool = False, **kwargs):
        self._use_feedback = use_feedback
        self._chat_history = chat_history

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Build user message with observation + task context
        content = []

        # Add 3 RGB views (left, front, right) as separate images
        obs = memory.current_observation
        if obs and obs.metadata:
            for view_name, meta_key in [("Left view", "left_rgb_path"),
                                         ("Front view", "front_rgb_path"),
                                         ("Right view", "right_rgb_path")]:
                path = obs.metadata.get(meta_key)
                if path:
                    image_url = _encode_image_base64(path)
                    content.append({"type": "text", "text": f"[{view_name}]"})
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif obs and obs.rgb_path:
            # Fallback: single image (e.g., testing without full bridge)
            image_url = _encode_image_base64(obs.rgb_path)
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        # Build text prompt
        text_parts = []

        # Task instruction
        text_parts.append(f"Task: {memory.task_description}")

        # Subtask progress from metadata
        if obs and obs.metadata:
            stage = obs.metadata.get("subtask_stage", "")
            total = obs.metadata.get("subtask_total", "")
            distance = obs.metadata.get("current_geo_distance", "")
            if stage and total:
                text_parts.append(f"Current subtask: {int(float(stage)) + 1}/{int(float(total))}")
            if distance:
                text_parts.append(f"Distance to current target: {float(distance):.1f}m")

        # Action history with feedback
        if self._use_feedback and memory.action_history:
            history_lines = []
            for action_name, feedback in memory.action_history[-10:]:  # last 10 steps
                history_lines.append(f"  {action_name} → {feedback}")
            text_parts.append("Recent actions:\n" + "\n".join(history_lines))

        # Available actions
        text_parts.append(f"Available actions: {', '.join(memory.action_space)}")
        text_parts.append('Respond with a JSON object: {"action": "<action_name>"}')

        content.append({"type": "text", "text": "\n\n".join(text_parts)})
        messages.append({"role": "user", "content": content})

        return messages

    def parse_response(self, llm_response: str, memory: AgentMemory) -> list[Action]:
        """Parse LLM response into a single Action."""
        try:
            # Try JSON parse
            response = json.loads(llm_response.strip())
            action_name = response.get("action", "move_forward")
        except (json.JSONDecodeError, AttributeError):
            # Fallback: look for action name in response text
            response_lower = llm_response.lower()
            if "stop" in response_lower:
                action_name = "stop"
            elif "turn_left" in response_lower or "left" in response_lower:
                action_name = "turn_left"
            elif "turn_right" in response_lower or "right" in response_lower:
                action_name = "turn_right"
            else:
                action_name = "move_forward"

        validated = validate_action_name(action_name, memory.action_space)
        return [Action(action_name=validated)]

    def get_response_format(self, memory: AgentMemory) -> dict:
        return {}
