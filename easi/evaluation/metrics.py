"""Metric aggregation utilities."""
from __future__ import annotations

from easi.core.episode import EpisodeRecord


def generic_aggregate(records: list[EpisodeRecord]) -> dict:
    """Compute generic metrics shared across all benchmarks.

    These are computed from per-episode results and included at the
    top level of summary.json alongside task-specific "metrics".

    Keys produced:
    - success_rate: mean of task_success (or success) across episodes
    - avg_steps: mean of num_steps across episodes
    - num_episodes: total count
    """
    if not records:
        return {}

    n = len(records)

    # Success: try task_success first, fall back to success
    successes = []
    for r in records:
        er = r.episode_results
        s = er.get("task_success", er.get("success"))
        if isinstance(s, (int, float)):
            successes.append(float(s))

    # Steps
    steps = []
    for r in records:
        er = r.episode_results
        s = er.get("num_steps", er.get("steps_taken"))
        if isinstance(s, (int, float)):
            steps.append(float(s))

    result = {"num_episodes": n}
    if successes:
        result["success_rate"] = round(sum(successes) / n, 4)
    if steps:
        result["avg_steps"] = round(sum(steps) / n, 1)
        sorted_steps = sorted(steps)
        result["median_steps"] = round(sorted_steps[len(sorted_steps) // 2], 1)

    return result


def default_aggregate(records: list[EpisodeRecord]) -> dict:
    """Default aggregation: average all numeric keys from episode_results.

    Used by tasks that don't override aggregate_results().

    Args:
        records: List of EpisodeRecord objects (one per episode).

    Returns:
        Summary metrics dict with avg_<key> for each numeric key.
    """
    if not records:
        return {}

    summary: dict = {}

    # Collect all numeric keys from episode_results
    numeric_keys: dict[str, list[float]] = {}
    for r in records:
        for key, value in r.episode_results.items():
            if isinstance(value, (int, float)):
                numeric_keys.setdefault(key, []).append(float(value))

    # Average each numeric metric over ALL episodes (not just those that emitted the key).
    # Failed episodes may not emit task-specific keys — they should contribute 0, not be excluded.
    total = len(records)
    for key, values in numeric_keys.items():
        summary[f"avg_{key}"] = round(sum(values) / total, 4)

    # Convenience aliases
    if "avg_success" in summary:
        summary["success_rate"] = summary["avg_success"]
    if "avg_task_success" in summary:
        summary["success_rate"] = summary["avg_task_success"]
    if "avg_num_steps" in summary:
        summary["avg_steps"] = summary["avg_num_steps"]

    return summary
