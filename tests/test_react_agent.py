"""Tests for the ReAct agent and PromptBuilder protocol."""
import json

import pytest

from easi.agents.prompt_builder import DefaultPromptBuilder, PromptBuilderProtocol
from easi.agents.react_agent import ReActAgent
from easi.core.episode import Action, Observation
from easi.core.memory import AgentMemory


class MockLLMClient:
    """Mock LLM that returns a fixed JSON plan."""
    def __init__(self, actions=None):
        self.actions = actions or [{"action": "MoveAhead"}]
        self.call_count = 0

    def generate(self, messages):
        self.call_count += 1
        return json.dumps({
            "observation": "I see a room.",
            "reasoning": "I should move forward.",
            "plan": "1. Move ahead",
            "executable_plan": self.actions,
        })


class TestDefaultPromptBuilder:
    def test_build_messages_returns_system_and_user(self):
        builder = DefaultPromptBuilder()
        memory = AgentMemory(
            task_description="Go to the kitchen.",
            action_space=["MoveAhead", "Stop"],
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        messages = builder.build_messages(memory)
        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert "MoveAhead" in messages[0]["content"]
        assert "JSON" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_build_messages_includes_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        builder = DefaultPromptBuilder()
        memory = AgentMemory(
            task_description="Go to the kitchen.",
            action_space=["MoveAhead", "Stop"],
            current_observation=Observation(rgb_path=str(img_path)),
        )
        messages = builder.build_messages(memory)
        content = messages[1]["content"]
        image_parts = [p for p in content if p["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_build_messages_with_history(self):
        builder = DefaultPromptBuilder()
        memory = AgentMemory(
            task_description="Go to the kitchen.",
            action_space=["MoveAhead", "TurnLeft", "Stop"],
            current_observation=Observation(rgb_path="/tmp/rgb.png"),
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        memory.record_step(obs, Action(action_name="MoveAhead"), llm_response="r1")
        memory.record_feedback("success")
        memory.record_step(obs, Action(action_name="TurnLeft"), llm_response="r2")
        memory.record_feedback("failed")
        messages = builder.build_messages(memory)
        text_content = ""
        for part in messages[1]["content"]:
            if part["type"] == "text":
                text_content += part["text"]
        assert "MoveAhead" in text_content
        assert "TurnLeft" in text_content

    def test_parse_response_valid(self):
        builder = DefaultPromptBuilder()
        memory = AgentMemory(action_space=["MoveAhead", "TurnLeft", "Stop"])
        response = json.dumps({
            "executable_plan": [{"action": "MoveAhead"}, {"action": "TurnLeft"}],
        })
        actions = builder.parse_response(response, memory)
        assert len(actions) == 2
        assert actions[0].action_name == "MoveAhead"
        assert actions[1].action_name == "TurnLeft"

    def test_parse_response_invalid_json(self):
        builder = DefaultPromptBuilder()
        memory = AgentMemory(action_space=["MoveAhead", "Stop"])
        actions = builder.parse_response("not json", memory)
        assert actions == []

    def test_parse_response_invalid_action(self):
        builder = DefaultPromptBuilder()
        memory = AgentMemory(action_space=["MoveAhead", "Stop"])
        response = json.dumps({
            "executable_plan": [{"action": "FlyToMoon"}],
        })
        actions = builder.parse_response(response, memory)
        assert actions == []


class CustomPromptBuilder:
    """A custom prompt builder for testing the delegation pattern."""
    def build_messages(self, memory):
        return [
            {"role": "system", "content": f"CUSTOM SYSTEM: {memory.task_description}"},
            {"role": "user", "content": f"CUSTOM STEP: steps={len(memory.steps)}"},
        ]

    def parse_response(self, llm_response, memory):
        data = json.loads(llm_response)
        return [Action(action_name=e["action"]) for e in data.get("executable_plan", [])]


class TestReActAgent:
    @pytest.fixture
    def agent(self):
        llm = MockLLMClient([{"action": "MoveAhead"}])
        return ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )

    @pytest.fixture
    def multi_action_agent(self):
        """Agent whose LLM returns a multi-action plan."""
        llm = MockLLMClient([
            {"action": "MoveAhead"},
            {"action": "TurnLeft"},
            {"action": "MoveAhead"},
        ])
        return ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )

    @pytest.fixture
    def custom_agent(self):
        llm = MockLLMClient([{"action": "MoveAhead"}])
        return ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
            prompt_builder=CustomPromptBuilder(),
        )

    def test_act_returns_action(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name == "MoveAhead"

    def test_calls_llm_once(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "Go to the goal.")
        assert agent.llm_client.call_count == 1

    def test_default_prompt_builder_used(self, agent):
        assert isinstance(agent.prompt_builder, DefaultPromptBuilder)

    def test_custom_prompt_builder(self, custom_agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        custom_agent.act(obs, "Go to the goal.")
        # Agent should have recorded the step in memory
        assert len(custom_agent.memory.steps) == 1

    # --- Action buffering tests ---

    def test_multi_action_buffer(self, multi_action_agent):
        """LLM returns 3 actions; first act() returns first, subsequent act()s
        return buffered actions WITHOUT calling LLM again."""
        obs = Observation(rgb_path="/tmp/rgb.png")

        a1 = multi_action_agent.act(obs, "Go to the goal.")
        assert a1.action_name == "MoveAhead"
        assert multi_action_agent.llm_client.call_count == 1

        multi_action_agent.add_feedback("MoveAhead", "success")
        a2 = multi_action_agent.act(obs, "Go to the goal.")
        assert a2.action_name == "TurnLeft"
        assert multi_action_agent.llm_client.call_count == 1

        multi_action_agent.add_feedback("TurnLeft", "success")
        a3 = multi_action_agent.act(obs, "Go to the goal.")
        assert a3.action_name == "MoveAhead"
        assert multi_action_agent.llm_client.call_count == 1

    def test_buffer_cleared_on_failure(self, multi_action_agent):
        """When add_feedback reports failure, buffer is cleared.
        Next act() re-queries LLM."""
        obs = Observation(rgb_path="/tmp/rgb.png")

        multi_action_agent.act(obs, "Go to the goal.")
        assert multi_action_agent.llm_client.call_count == 1

        multi_action_agent.add_feedback("MoveAhead", "failed: obstacle ahead")

        multi_action_agent.act(obs, "Go to the goal.")
        assert multi_action_agent.llm_client.call_count == 2

    def test_buffer_empty_after_all_consumed(self, multi_action_agent):
        """After all buffered actions consumed, next act() queries LLM."""
        obs = Observation(rgb_path="/tmp/rgb.png")

        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("MoveAhead", "success")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("TurnLeft", "success")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("MoveAhead", "success")
        assert multi_action_agent.llm_client.call_count == 1

        multi_action_agent.act(obs, "Go to the goal.")
        assert multi_action_agent.llm_client.call_count == 2

    def test_parse_error_returns_stop(self):
        """When LLM returns invalid JSON, agent returns Stop."""
        llm = type('MockLLM', (), {'generate': lambda self, m: 'not json at all'})()
        agent = ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name == "Stop"
        assert not agent._action_buffer

    def test_invalid_action_name_fallback(self):
        """When LLM returns action not in action_space, fallback to Stop."""
        llm = MockLLMClient([{"action": "FlyToMoon"}])
        agent = ReActAgent(
            llm_client=llm,
            action_space=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
        )
        obs = Observation(rgb_path="/tmp/rgb.png")
        action = agent.act(obs, "Go to the goal.")
        assert action.action_name == "Stop"

    def test_reset_clears_buffer_and_memory(self, multi_action_agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.reset()
        assert len(multi_action_agent.memory.steps) == 0
        assert len(multi_action_agent._action_buffer) == 0

    def test_memory_records_steps(self, agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        agent.act(obs, "Go to the goal.")
        assert len(agent.memory.steps) == 1
        assert agent.memory.steps[0].action.action_name == "MoveAhead"
        assert agent.memory.steps[0].llm_response is not None

    def test_buffered_steps_have_no_llm_response(self, multi_action_agent):
        obs = Observation(rgb_path="/tmp/rgb.png")
        multi_action_agent.act(obs, "Go to the goal.")
        multi_action_agent.add_feedback("MoveAhead", "success")
        multi_action_agent.act(obs, "Go to the goal.")  # buffered
        assert multi_action_agent.memory.steps[1].llm_response is None
