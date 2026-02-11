"""Core data structures for observations, actions, and step results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """Observation produced by a simulator after reset or step."""

    rgb_path: str
    depth_path: str | None = None
    agent_pose: list[float] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Action:
    """An action to be executed in the simulator."""

    action_name: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of executing one step in the simulator."""

    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: dict[str, float] = field(default_factory=dict)
