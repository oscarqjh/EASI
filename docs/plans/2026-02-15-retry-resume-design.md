# Retry & Resume Design

## Goal

Make evaluation runs resilient to transient LLM errors (timeouts, rate limits) and resumable after crashes.

## Feature 1: Adaptive Retry

Use LiteLLM's built-in `num_retries` parameter. LiteLLM uses tenacity internally for exponential backoff with jitter, and retries on `Timeout`, `RateLimitError`, `APIConnectionError`, and `InternalServerError`.

### LLMClient

`LLMClient.__init__` gains a `num_retries` parameter (default 3). It's passed through to `litellm.completion()`:

```python
class LLMClient:
    def __init__(self, model, base_url=None, num_retries=3, **kwargs):
        self.num_retries = num_retries
        ...

    def generate(self, messages, response_format=None):
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "num_retries": self.num_retries,
            **self.default_kwargs,
        }
        ...
```

### CLI

Add `--max-retries` (default 3) to `easi run`. Passed through: CLI -> `EvaluationRunner` -> `LLMClient(num_retries=...)`.

### EvaluationRunner

`__init__` gains `max_retries: int = 3`. In `_create_agent()`, passes it to `LLMClient`.

## Feature 2: Resume

Allow `--resume <run_dir>` to continue a crashed run from where it left off.

### How it works

1. **Load existing config**: Read `config.json` from the resume dir. Validate `task_name` matches.
2. **Scan completed episodes**: Glob `episodes/*/result.json`. Each existing `result.json` = completed episode. Parse them into `all_results`.
3. **Re-run last completed episode**: The last episode may have been cut off mid-way. Pop it from `all_results`, clear its directory (delete `result.json`, `trajectory.jsonl`, `step_*.png`), and re-run it from scratch.
4. **Skip earlier episodes**: Start the episode loop from `len(all_results)` instead of 0.
5. **Reuse run_dir**: Don't create a new `run_id`. Write to the same directory.

### _load_completed_results

```python
def _load_completed_results(self, run_dir: Path) -> list[dict]:
    episodes_dir = run_dir / "episodes"
    result_files = sorted(episodes_dir.glob("*/result.json"))

    if not result_files:
        return []

    # Load all completed results except the last
    all_results = []
    for rf in result_files[:-1]:
        all_results.append(json.loads(rf.read_text()))

    # Clean the last episode directory for re-run
    last_episode_dir = result_files[-1].parent
    for f in last_episode_dir.iterdir():
        f.unlink()

    return all_results
```

### run() changes

```python
def run(self, max_episodes=None):
    if self.resume_dir:
        run_dir = self.resume_dir
        all_results = self._load_completed_results(run_dir)
        start_index = len(all_results)
    else:
        run_dir = self.output_dir / self.task_name / self.run_id
        all_results = []
        start_index = 0

    ...
    for i, episode in enumerate(episodes):
        if i < start_index:
            continue
        ...
```

### CLI

Add `--resume <path>` to `easi run`. Passed to `EvaluationRunner(resume_dir=...)`.

### Example

```bash
# First run — crashes at episode 15
easi run ebalfred_base --agent react --backend openai --model gpt-4o

# Resume — skips episodes 0-13, re-runs 14 (cleared), continues 15-49
easi run ebalfred_base --agent react --backend openai --model gpt-4o \
    --resume logs/ebalfred_base/20260215_164200/
```

## Files Changed

| File | Change |
|---|---|
| `easi/llm/client.py` | Add `num_retries` param, pass to `litellm.completion()` |
| `easi/evaluation/runner.py` | Add `max_retries`, `resume_dir` params. Add `_load_completed_results()`. Modify `run()` for resume. |
| `easi/cli.py` | Add `--max-retries` (default 3) and `--resume` (path) flags |
| `tests/test_llm_client.py` | Test `num_retries` pass-through |
| `tests/test_e2e_evaluation.py` | Test resume: run 3 episodes, simulate crash, resume skips completed, re-runs last |

## Data Flow

```
CLI --max-retries 3 --resume logs/ebalfred_base/20260215_164200/
    |
    v
EvaluationRunner(max_retries=3, resume_dir=Path(...))
    |
    +---> _load_completed_results(run_dir)
    |         scan episodes/*/result.json
    |         load results [0..N-2]
    |         clear episode dir [N-1]
    |         return results, start_index = N-1
    |
    +---> _create_agent(...)
    |         LLMClient(num_retries=3)
    |
    +---> episode loop: skip i < start_index, run rest normally
```
