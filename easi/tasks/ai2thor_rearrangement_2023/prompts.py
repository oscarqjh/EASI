"""Prompt builder for AI2-THOR Rearrangement task.

Constructs multi-modal prompts with observation images, GPS data,
action history, and structured JSON output schema.

For the 1-phase track, each step includes both the current (shuffled)
observation and a goal (walkthrough) image from the same viewpoint.
"""
from __future__ import annotations

import json

from easi.agents.prompt_builder import _encode_image_base64, validate_action_name
from easi.core.episode import Action
from easi.core.memory import AgentMemory
from easi.utils.json_repair import fix_json
from easi.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are an embodied AI agent performing an object rearrangement task in a 3D indoor environment (AI2-THOR).

## Goal
{instruction}

## Observations
You receive two images at each step:
1. **Current scene** — what the environment looks like RIGHT NOW (shuffled state)
2. **Goal scene** — what the environment SHOULD look like from the same viewpoint (target state)

Compare these two images to identify which objects are misplaced and where they should go.

## Available Actions
{action_list}

## Action Types
- **Navigation**: move_ahead (0.25m step), move_left/right/back, rotate_left/right (90°), look_up/look_down (30°), stand/crouch
- **Pickup**: pickup_<object_type> — picks up the nearest visible object of that type (must have empty hands)
- **Drop**: drop_held_object_with_snap — drops held object, snapping to goal position if close enough
- **Open/Close**: open_by_type_<type> — toggles openness of nearest visible cabinet/drawer/fridge/etc.
- **Done**: Signal that all objects are in their correct positions

## Strategy
1. Compare the current and goal images to identify misplaced objects
2. Navigate to a misplaced object
3. Pick it up with the matching pickup_<type> action
4. Navigate toward where it belongs (compare with goal image)
5. Use drop_held_object_with_snap — it snaps the object to its goal if you're close
6. Repeat for remaining misplaced objects
7. Say "done" when finished

## Rules
- You can hold only ONE object at a time
- drop_held_object_with_snap works best when the goal location is visible
- If pickup fails, try moving closer or adjusting your view angle
- Grid movement: 0.25m steps, 90° rotations
"""

OUTPUT_SCHEMA = """\
Respond in this exact JSON format:
```json
{{
  "observation": "describe what you see in current vs goal images",
  "reasoning": "what to do next and why",
  "plan": [
    {{"action_name": "<name>"}},
    ...
  ]
}}
```
Plan 1-5 actions. Use exact action names from the list above."""


class AI2THORRearrangement2023PromptBuilder:
    """Prompt builder for the rearrangement task."""

    def __init__(
        self,
        use_feedback: bool = True,
        chat_history: bool = True,
        message_window_len: int = 10,
        **kwargs,
    ):
        self.use_feedback = use_feedback
        self.chat_history = chat_history
        self.message_window_len = message_window_len
        self._action_list_str = ""
        self._action_name_set: set[str] = set()

    def set_action_space(self, actions: list[str]):
        self._action_name_set = set(actions)
        parts = [f"{i}: {a}" for i, a in enumerate(actions)]
        self._action_list_str = ", ".join(parts)

    def build_messages(self, memory: AgentMemory) -> list[dict]:
        # Lazy-init action space from memory (set by agent constructor)
        if not self._action_list_str and memory.action_space:
            self.set_action_space(memory.action_space)

        instruction = memory.task_description or "Rearrange objects to match the goal."

        system_text = SYSTEM_PROMPT.format(
            instruction=instruction,
            action_list=self._action_list_str,
        )

        messages = [{"role": "system", "content": system_text}]

        # Chat history (if enabled)
        if self.chat_history and memory.steps:
            window = memory.steps[-self.message_window_len:]
            for step in window:
                user_content = self._make_history_content(step)
                messages.append({"role": "user", "content": user_content})
                if step.llm_response:
                    messages.append({
                        "role": "assistant",
                        "content": step.llm_response,
                    })

        # Current turn
        user_content = self._make_current_content(memory)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _make_history_content(self, step) -> list[dict]:
        content = []
        metadata = step.observation.metadata if step.observation else {}

        # Current observation image
        if step.observation and step.observation.rgb_path:
            img_url = _encode_image_base64(step.observation.rgb_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })

        # Goal image (walkthrough state from same viewpoint)
        goal_path = metadata.get("goal_rgb_path")
        if goal_path:
            img_url = _encode_image_base64(goal_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })

        text = "Observation (image 1: current, image 2: goal)."
        if self.use_feedback and step.feedback:
            text += f"\nFeedback: {step.feedback}"

        # GPS overlay from observation metadata
        gps_text = self._format_gps(metadata)
        if gps_text:
            text += f"\n{gps_text}"

        content.append({"type": "text", "text": text})
        return content

    def _make_current_content(self, memory: AgentMemory) -> list[dict]:
        content = []
        metadata = (
            memory.current_observation.metadata
            if memory.current_observation else {}
        )

        # Current observation image
        if memory.current_observation and memory.current_observation.rgb_path:
            img_url = _encode_image_base64(memory.current_observation.rgb_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })

        # Goal image (walkthrough state from same viewpoint)
        goal_path = metadata.get("goal_rgb_path")
        if goal_path:
            img_url = _encode_image_base64(goal_path)
            if img_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url},
                })

        has_goal = bool(goal_path)
        img_label = " (image 1: current, image 2: goal)" if has_goal else ""

        if memory.is_first_turn:
            text = f"First observation{img_label}. Begin the rearrangement task."
        else:
            text = f"Current observation{img_label}."
            if self.use_feedback and memory.steps:
                last = memory.steps[-1]
                if last.feedback:
                    text += f"\nFeedback: {last.feedback}"

        # GPS and held-object from last step's observation metadata
        if memory.steps:
            last_metadata = memory.steps[-1].observation.metadata if memory.steps[-1].observation else {}
            gps_text = self._format_gps(last_metadata)
            if gps_text:
                text += f"\n{gps_text}"
            held = last_metadata.get("held_object", "none")
            text += f"\nHolding: {held}"

        text += f"\n\n{OUTPUT_SCHEMA}"
        content.append({"type": "text", "text": text})
        return content

    def _format_gps(self, metadata: dict | None) -> str:
        if not metadata:
            return ""
        parts = []
        if "agent_x" in metadata:
            x = float(metadata["agent_x"])
            y = float(metadata["agent_y"])
            z = float(metadata["agent_z"])
            parts.append(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
        if "agent_rotation" in metadata:
            parts.append(f"Rotation: {float(metadata['agent_rotation']):.0f}")
        if "agent_horizon" in metadata:
            parts.append(f"Horizon: {float(metadata['agent_horizon']):.0f}")
        return "GPS: " + ", ".join(parts) if parts else ""

    def parse_response(
        self, llm_response: str, memory: AgentMemory
    ) -> list[Action]:
        try:
            fixed = fix_json(llm_response)
            data = json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse LLM response as JSON")
            return [Action(action_name="done")]

        plan = data.get("plan", [])
        if not plan:
            return [Action(action_name="done")]

        actions = []
        for entry in plan:
            name = entry.get("action_name", "")
            if name in self._action_name_set:
                actions.append(Action(action_name=name))
            else:
                logger.warning("Unknown action '%s', stopping plan", name)
                break

        return actions if actions else [Action(action_name="done")]
