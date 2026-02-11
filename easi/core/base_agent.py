"""Abstract base class for agents.

An agent bridges the LLM inference endpoint and the simulator. It manages:
- Chat history across steps
- Prompt building (system prompt, per-step observation prompts)
- Parsing LLM text responses into structured Action objects

Concrete agents subclass this and implement the three abstract methods:
_build_system_prompt, _build_step_prompt, _parse_action.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from easi.core.episode import Action, Observation
from easi.core.exceptions import ActionParseError
from easi.core.protocols import LLMClientProtocol

logger = logging.getLogger("easi.core.base_agent")


class BaseAgent(ABC):
    """Abstract base for agents that bridge LLM inference and simulator actions."""

    def __init__(self, llm_client: LLMClientProtocol | None, action_space: list[str]):
        self.llm_client = llm_client
        self.action_space = action_space
        self._chat_history: list[dict[str, str]] = []
        self._step_count: int = 0

    @abstractmethod
    def _build_system_prompt(self, task_description: str) -> str:
        """Build the system prompt describing the task and action space."""
        ...

    @abstractmethod
    def _build_step_prompt(self, observation: Observation) -> str:
        """Build the user message for a single step (includes observation description)."""
        ...

    @abstractmethod
    def _parse_action(self, llm_response: str) -> Action:
        """Parse the LLM's text response into a structured Action.

        Raises:
            ActionParseError: If the response cannot be parsed.
        """
        ...

    def reset(self) -> None:
        """Clear chat history for a new episode."""
        self._chat_history = []
        self._step_count = 0

    def act(self, observation: Observation, task_description: str) -> Action:
        """Full agent loop for one step.

        1. Build system prompt on first call
        2. Build step prompt with observation
        3. Collect image paths from observation
        4. Call LLM inference endpoint
        5. Parse response into Action
        6. Append to chat history
        """
        # Add system prompt on first step
        if self._step_count == 0:
            system_prompt = self._build_system_prompt(task_description)
            if system_prompt:
                self._chat_history.append({
                    "role": "system",
                    "content": system_prompt,
                })

        # Build step prompt
        self._step_count += 1
        step_prompt = self._build_step_prompt(observation)

        # Collect images
        images = []
        if observation.rgb_path:
            images.append(observation.rgb_path)

        # Add user message to history
        user_message = {"role": "user", "content": step_prompt}
        if images:
            user_message["images"] = images
        self._chat_history.append(user_message)

        # Call LLM
        if self.llm_client is not None:
            llm_response = self.llm_client.generate(
                messages=self._chat_history,
                images=images if images else None,
            )
        else:
            llm_response = ""

        # Parse action
        action = self._parse_action(llm_response)

        # Append assistant response to history
        self._chat_history.append({
            "role": "assistant",
            "content": llm_response,
        })

        logger.debug(
            "Step %d: action=%s params=%s",
            self._step_count,
            action.action_name,
            action.params,
        )

        return action

    @property
    def chat_history(self) -> list[dict[str, str]]:
        """Return a copy of the chat history."""
        return self._chat_history.copy()
