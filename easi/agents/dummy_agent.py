"""Dummy agent for testing — returns random actions without calling an LLM."""

from __future__ import annotations

import random

from easi.core.base_agent import BaseAgent
from easi.core.episode import Action, Observation


class DummyAgent(BaseAgent):
    """Agent that picks random actions from the action space.

    Does not call the LLM client. Useful for testing the full pipeline
    without needing a running LLM server.
    """

    def __init__(self, action_space: list[str], seed: int | None = None):
        super().__init__(llm_client=None, action_space=action_space)
        self._rng = random.Random(seed)

    def _build_system_prompt(self, task_description: str) -> str:
        return ""  # no system prompt needed

    def _build_step_prompt(self, observation: Observation) -> str:
        return "Choose an action."

    def _parse_action(self, llm_response: str) -> Action:
        """Pick a random action from the action space."""
        action_name = self._rng.choice(self.action_space)
        return Action(action_name=action_name)

    def act(self, observation: Observation, task_description: str) -> Action:
        """Override to skip LLM call entirely."""
        self._step_count += 1

        action = self._parse_action("")

        # Still maintain chat history for compatibility
        self._chat_history.append({
            "role": "user",
            "content": f"Step {self._step_count}. Observation at {observation.rgb_path}",
        })
        self._chat_history.append({
            "role": "assistant",
            "content": f"Action: {action.action_name}",
        })

        return action
