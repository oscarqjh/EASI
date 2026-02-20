# Retry & Resume Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make evaluation runs resilient to transient LLM errors and resumable after crashes.

**Architecture:** Two independent features wired through CLI → EvaluationRunner → LLMClient. Retry uses LiteLLM's built-in `num_retries` parameter (exponential backoff with jitter). Resume scans existing `result.json` files to skip completed episodes.

**Tech Stack:** Python, LiteLLM (`num_retries`), pytest, argparse

**Design doc:** `docs/plans/2026-02-15-retry-resume-design.md`

---

### Task 1: Add `num_retries` to LLMClient

**Files:**
- Modify: `easi/llm/client.py:39-47` (constructor), `easi/llm/client.py:59-63` (generate call_kwargs)
- Test: `tests/test_llm_client.py`

**Step 1: Write the failing tests**

Add to `tests/test_llm_client.py`:

```python
class TestLLMClientRetries:
    def test_default_num_retries(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/gpt-4o")
        assert client.num_retries == 3

    def test_custom_num_retries(self):
        from easi.llm.client import LLMClient
        client = LLMClient(model="openai/gpt-4o", num_retries=5)
        assert client.num_retries == 5

    @patch("easi.llm.client.litellm")
    def test_num_retries_passed_to_litellm(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        client = LLMClient(model="openai/gpt-4o", num_retries=7)
        client.generate([{"role": "user", "content": "hi"}])

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("num_retries") == 7

    @patch("easi.llm.client.litellm")
    def test_zero_retries_passed_through(self, mock_litellm):
        from easi.llm.client import LLMClient

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response
        mock_litellm.completion_cost.return_value = 0.0

        client = LLMClient(model="openai/gpt-4o", num_retries=0)
        client.generate([{"role": "user", "content": "hi"}])

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("num_retries") == 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_llm_client.py::TestLLMClientRetries -v`
Expected: FAIL — `LLMClient` has no `num_retries` attribute

**Step 3: Implement**

In `easi/llm/client.py`, modify the constructor (lines 39-47):

```python
def __init__(
    self,
    model: str,
    base_url: str | None = None,
    num_retries: int = 3,
    **kwargs: Any,
):
    self.model = model
    self.base_url = base_url
    self.num_retries = num_retries
    self.default_kwargs = kwargs
    self._usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "num_calls": 0,
        "cost_usd": 0.0,
    }
```

In `generate()` (lines 59-63), add `num_retries` to `call_kwargs`:

```python
call_kwargs: dict[str, Any] = {
    "model": self.model,
    "messages": messages,
    "num_retries": self.num_retries,
    **self.default_kwargs,
}
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_llm_client.py -v`
Expected: ALL PASS (existing + new)

**Step 5: Commit**

```bash
git add easi/llm/client.py tests/test_llm_client.py
git commit -m "feat: add num_retries to LLMClient for LiteLLM retry support"
```

---

### Task 2: Wire `max_retries` through EvaluationRunner

**Files:**
- Modify: `easi/evaluation/runner.py:47-69` (constructor), `easi/evaluation/runner.py:338-342` (LLMClient instantiation)
- Test: `tests/test_e2e_evaluation.py`

**Step 1: Write the failing test**

Add to `tests/test_e2e_evaluation.py`:

```python
def test_max_retries_stored(self, tmp_path):
    """Verify max_retries is stored and appears in config."""
    output_dir = tmp_path / "logs"
    runner = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        max_retries=5,
    )
    assert runner.max_retries == 5
    runner.run(max_episodes=1)

    run_dir = _find_run_dir(output_dir)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["cli_options"]["max_retries"] == 5
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py::TestE2EEvaluation::test_max_retries_stored -v`
Expected: FAIL — `EvaluationRunner.__init__()` got unexpected keyword `max_retries`

**Step 3: Implement**

In `easi/evaluation/runner.py`, add `max_retries` to constructor (line 59, after `port`):

```python
def __init__(
    self,
    task_name: str,
    agent_type: str = "dummy",
    output_dir: Path | str = "./logs",
    data_dir: Path | str = "./datasets",
    llm_base_url: str | None = None,
    agent_seed: int | None = None,
    backend: str | None = None,
    model: str = "default",
    port: int = 8080,
    llm_kwargs_raw: str | None = None,
    max_retries: int = 3,
):
    ...
    self.max_retries = max_retries
    ...
```

Add `max_retries` to the config dict in `run()` (inside `cli_options`):

```python
"max_retries": self.max_retries,
```

In `_create_agent()`, pass `num_retries` to `LLMClient` (around line 338):

```python
llm = LLMClient(
    model=litellm_model,
    base_url=base_url,
    num_retries=self.max_retries,
    **client_kwargs,
)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add easi/evaluation/runner.py tests/test_e2e_evaluation.py
git commit -m "feat: wire max_retries through EvaluationRunner to LLMClient"
```

---

### Task 3: Add `--max-retries` CLI flag

**Files:**
- Modify: `easi/cli.py:80-98` (run_parser args), `easi/cli.py:288-303` (cmd_run)

**Step 1: Write the failing test**

Add to `tests/test_e2e_evaluation.py` (or create a quick inline test):

This is a CLI integration test. Since the existing tests are E2E with `EvaluationRunner`, we can verify the CLI parser separately:

```python
def test_cli_max_retries_default():
    """CLI --max-retries defaults to 3."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["run", "dummy_task"])
    assert args.max_retries == 3

def test_cli_max_retries_custom():
    """CLI --max-retries is parseable."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["run", "dummy_task", "--max-retries", "5"])
    assert args.max_retries == 5
```

Add these to a new `TestCLIParsing` class in `tests/test_e2e_evaluation.py`.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py::TestCLIParsing -v`
Expected: FAIL — `args` has no attribute `max_retries`

**Step 3: Implement**

In `easi/cli.py`, add after line 98 (`--llm-kwargs`):

```python
run_parser.add_argument("--max-retries", type=int, default=3,
                        help="Max LLM retry attempts on transient errors (default: 3)")
```

In `cmd_run` (line 288), add `max_retries` parameter and pass to `EvaluationRunner`:

```python
def cmd_run(task_name, agent_type, output_dir, data_dir, max_episodes,
            llm_url, seed, backend, model, port, llm_kwargs_raw, max_retries):
    from easi.evaluation.runner import EvaluationRunner

    runner = EvaluationRunner(
        task_name=task_name,
        agent_type=agent_type,
        output_dir=output_dir,
        data_dir=data_dir,
        llm_base_url=llm_url,
        agent_seed=seed,
        backend=backend,
        model=model,
        port=port,
        llm_kwargs_raw=llm_kwargs_raw,
        max_retries=max_retries,
    )
    ...
```

In `main()` dispatch (line 361), add `args.max_retries`:

```python
cmd_run(args.task, args.agent, args.output_dir, args.data_dir,
        args.max_episodes, args.llm_url, args.seed,
        args.backend, args.model, args.port, args.llm_kwargs,
        args.max_retries)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add easi/cli.py tests/test_e2e_evaluation.py
git commit -m "feat: add --max-retries CLI flag for LLM retry control"
```

---

### Task 4: Add `--resume` CLI flag and `resume_dir` to EvaluationRunner

**Files:**
- Modify: `easi/cli.py:98-99` (run_parser args), `easi/cli.py:288-303` (cmd_run)
- Modify: `easi/evaluation/runner.py:47-70` (constructor)

**Step 1: Write the failing tests**

Add to `TestCLIParsing` in `tests/test_e2e_evaluation.py`:

```python
def test_cli_resume_default():
    """CLI --resume defaults to None."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["run", "dummy_task"])
    assert args.resume is None

def test_cli_resume_custom():
    """CLI --resume accepts a path."""
    from easi.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["run", "dummy_task", "--resume", "/tmp/logs/run_123"])
    assert args.resume == "/tmp/logs/run_123"
```

Add to `TestE2EEvaluation`:

```python
def test_resume_dir_stored(self, tmp_path):
    """Verify resume_dir is stored."""
    runner = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=tmp_path / "logs",
        resume_dir=tmp_path / "old_run",
    )
    assert runner.resume_dir == tmp_path / "old_run"
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py::TestCLIParsing::test_cli_resume_default tests/test_e2e_evaluation.py::TestCLIParsing::test_cli_resume_custom tests/test_e2e_evaluation.py::TestE2EEvaluation::test_resume_dir_stored -v`
Expected: FAIL

**Step 3: Implement**

In `easi/cli.py`, add after `--max-retries`:

```python
run_parser.add_argument("--resume", type=str, default=None,
                        help="Path to a previous run directory to resume from")
```

In `easi/evaluation/runner.py` constructor, add `resume_dir`:

```python
def __init__(
    self,
    ...
    max_retries: int = 3,
    resume_dir: Path | str | None = None,
):
    ...
    self.max_retries = max_retries
    self.resume_dir = Path(resume_dir) if resume_dir else None
    ...
```

In `easi/cli.py` `cmd_run`, add `resume` parameter and pass to `EvaluationRunner`:

```python
def cmd_run(task_name, agent_type, output_dir, data_dir, max_episodes,
            llm_url, seed, backend, model, port, llm_kwargs_raw,
            max_retries, resume):
    ...
    runner = EvaluationRunner(
        ...
        max_retries=max_retries,
        resume_dir=resume,
    )
    ...
```

In `main()` dispatch, add `args.resume`:

```python
cmd_run(args.task, args.agent, args.output_dir, args.data_dir,
        args.max_episodes, args.llm_url, args.seed,
        args.backend, args.model, args.port, args.llm_kwargs,
        args.max_retries, args.resume)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add easi/cli.py easi/evaluation/runner.py tests/test_e2e_evaluation.py
git commit -m "feat: add --resume CLI flag and resume_dir to EvaluationRunner"
```

---

### Task 5: Implement `_load_completed_results()` and resume logic in `run()`

**Files:**
- Modify: `easi/evaluation/runner.py:94-185` (run method + new helper)
- Test: `tests/test_e2e_evaluation.py`

**Step 1: Write the failing tests**

Add to `TestE2EEvaluation` in `tests/test_e2e_evaluation.py`:

```python
def test_resume_skips_completed_episodes(self, tmp_path):
    """Resume skips completed episodes and re-runs the last one."""
    output_dir = tmp_path / "logs"

    # First run: complete 3 episodes
    runner1 = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        agent_seed=42,
    )
    results1 = runner1.run()
    assert len(results1) == 3

    run_dir = _find_run_dir(output_dir)

    # Resume: should re-run episode 2 (last), skip 0 and 1
    runner2 = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        resume_dir=run_dir,
        agent_seed=42,
    )
    results2 = runner2.run()
    assert len(results2) == 3  # 2 loaded + 1 re-run (episode 2)

def test_resume_clears_last_episode_dir(self, tmp_path):
    """Resume clears the last episode directory before re-running."""
    output_dir = tmp_path / "logs"

    runner1 = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        agent_seed=42,
    )
    runner1.run()

    run_dir = _find_run_dir(output_dir)
    episodes_dir = run_dir / "episodes"
    episode_dirs = sorted(episodes_dir.iterdir())
    last_ep_dir = episode_dirs[-1]

    # Add a marker file to the last episode dir
    marker = last_ep_dir / "marker.txt"
    marker.write_text("should be deleted")

    runner2 = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        resume_dir=run_dir,
        agent_seed=42,
    )
    runner2.run()

    # Marker should be gone (dir was cleared)
    assert not marker.exists()
    # But result.json should exist (re-run completed)
    assert (last_ep_dir / "result.json").exists()

def test_resume_produces_valid_summary(self, tmp_path):
    """Resumed run produces a valid summary with all episodes."""
    output_dir = tmp_path / "logs"

    runner1 = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        agent_seed=42,
    )
    runner1.run()

    run_dir = _find_run_dir(output_dir)

    runner2 = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        resume_dir=run_dir,
        agent_seed=42,
    )
    results2 = runner2.run()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["num_episodes"] == 3

def test_resume_empty_run_dir(self, tmp_path):
    """Resume with no completed episodes runs all from scratch."""
    output_dir = tmp_path / "logs"
    run_dir = output_dir / "dummy_task" / "fake_run"
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True)

    # Write a minimal config.json so resume can read it
    import json
    config = {"run_id": "fake_run", "cli_options": {"task_name": "dummy_task"}}
    (run_dir / "config.json").write_text(json.dumps(config))

    runner = EvaluationRunner(
        task_name="dummy_task",
        agent_type="dummy",
        output_dir=output_dir,
        resume_dir=run_dir,
        agent_seed=42,
    )
    results = runner.run()
    assert len(results) == 3  # All episodes run from scratch
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py::TestE2EEvaluation::test_resume_skips_completed_episodes -v`
Expected: FAIL — resume logic not implemented

**Step 3: Implement**

Add `_load_completed_results()` method to `EvaluationRunner`:

```python
def _load_completed_results(self, run_dir: Path) -> list[dict]:
    """Load results from a previous run for resume.

    Returns results from all completed episodes except the last one.
    The last episode directory is cleared for re-run (it may have been
    interrupted mid-way).
    """
    episodes_dir = run_dir / "episodes"
    if not episodes_dir.exists():
        return []

    result_files = sorted(episodes_dir.glob("*/result.json"))
    if not result_files:
        return []

    # Load all completed results except the last
    all_results = []
    for rf in result_files[:-1]:
        all_results.append(json.loads(rf.read_text()))

    # Clear the last episode directory for re-run
    last_episode_dir = result_files[-1].parent
    for f in last_episode_dir.iterdir():
        f.unlink()

    return all_results
```

Modify `run()` to support resume — change the beginning of the method:

```python
def run(self, max_episodes: int | None = None) -> list[dict]:
    """Run evaluation and return per-episode metric dicts."""
    if self.resume_dir:
        run_dir = self.resume_dir
        all_results = self._load_completed_results(run_dir)
        start_index = len(all_results)
        logger.info(
            "Resuming from %s — %d completed episodes, starting from %d",
            run_dir, len(all_results), start_index,
        )
    else:
        run_dir = self.output_dir / self.task_name / self.run_id
        all_results = []
        start_index = 0

    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load task
    task = self._create_task()
    episodes = task.load_episodes()
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    ...  # rest stays the same until the episode loop

    for i, episode in enumerate(episodes):
        if i < start_index:
            continue
        ...  # existing episode loop body
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_e2e_evaluation.py -v`
Expected: ALL PASS

**Step 5: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v --timeout=60`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add easi/evaluation/runner.py tests/test_e2e_evaluation.py
git commit -m "feat: implement resume logic — scan completed episodes, re-run last"
```

---

## Dependency Graph

```
Task 1 (LLMClient num_retries)
    └──> Task 2 (Runner max_retries wiring)
             └──> Task 3 (CLI --max-retries)

Task 4 (CLI --resume + resume_dir)
    └──> Task 5 (Resume logic in run())
```

Tasks 1-3 (retry) and Tasks 4-5 (resume) are independent chains.
Tasks 1+4 can run in parallel. Tasks 2+5 can run in parallel (after 1 and 4 respectively).

## Verification

```bash
# Full test suite
.venv/bin/pytest tests/ -v --timeout=60

# Quick CLI smoke test
.venv/bin/easi run dummy_task --agent dummy --max-retries 5
.venv/bin/easi run dummy_task --agent dummy --resume logs/dummy_task/<run_id>/
```

## Files Changed Summary

| File | Change |
|---|---|
| `easi/llm/client.py` | Add `num_retries` param (default 3), pass to `litellm.completion()` |
| `easi/evaluation/runner.py` | Add `max_retries` + `resume_dir` params, `_load_completed_results()`, resume logic in `run()` |
| `easi/cli.py` | Add `--max-retries` (default 3) and `--resume` (path) CLI flags |
| `tests/test_llm_client.py` | 4 new tests for `num_retries` |
| `tests/test_e2e_evaluation.py` | ~8 new tests for retry config + resume |
