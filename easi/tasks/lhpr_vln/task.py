"""LHPR-VLN task for EASI.

Adapts the LHPR-VLN benchmark to EASI's task interface.
Episodes are loaded from preprocessed JSONL files (data/{split}.jsonl)
with custom split names (val, test, unseen_val, unseen_test).
Each episode contains 2-4 sequential navigation subtasks in HM3D scenes.

Metrics:
- Per-episode: task_success, oracle_success, spl, navigation_error,
  isr, csr, cgt, tar (computed per-episode for result.json)
- Aggregate: All 8 official metrics + contest_score
  (using vendored NavigationMetrics)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from easi.core.base_task import BaseTask
from easi.core.episode import EpisodeRecord, StepResult
from easi.tasks.lhpr_vln.actions import get_action_space
from easi.tasks.lhpr_vln.vendor.metrics import NavigationMetrics
from easi.utils.logging import get_logger

logger = get_logger(__name__)


class LHPRVLNTask(BaseTask):

    def _build_action_space(self) -> list[str]:
        return get_action_space()

    def _load_episodes_from_hf(self, dataset_config: dict) -> list[dict]:
        """Load episodes from HF dataset with custom split names.

        HF auto-detection merges val/unseen_val into a single 'validation'
        split. We bypass this by loading the specific JSONL file directly.
        """
        from datasets import load_dataset
        from easi.core.base_task import hf_row_to_episode

        data_dir = self.download_dataset()
        split_name = dataset_config.get("split")
        data_file = str(data_dir / "data" / f"{split_name}.jsonl")

        logger.info("Loading episodes from %s (split=%s)", data_file, split_name)

        hf_cache = Path(tempfile.gettempdir()) / "easi_hf_cache"
        ds = load_dataset(
            "json", data_files=data_file, split="train",
            cache_dir=str(hf_cache),
        )
        episodes = [hf_row_to_episode(row) for row in ds]

        for ep in episodes:
            ep["_data_dir"] = str(data_dir)

        logger.info("Loaded %d episodes (split=%s)", len(episodes), split_name)
        return episodes

    def get_task_yaml_path(self) -> Path:
        return Path(__file__).parent / "_base.yaml"

    def get_bridge_script_path(self) -> Path:
        return Path(__file__).parent / "bridge.py"

    def get_instruction(self, episode: dict) -> str:
        return episode.get("instruction", self.name)

    def format_reset_config(self, episode: dict) -> dict:
        """Map LHPR-VLN episode dict to bridge reset config.

        Episode keys come from the preprocessed JSONL (lowercase):
            id, instruction, scene, robot, objects, regions, rooms,
            gt_steps, subtask_list, num_targets, batch
        """
        return {
            "episode_id": episode.get("id", "unknown"),
            "scene_id": episode["scene"],
            "robot": episode.get("robot", "spot"),
            "instruction": episode.get("instruction", ""),
            "targets": episode["objects"],
            "regions": episode["regions"],
            "gt_step": episode.get("gt_steps", []),
            "data_dir": episode.get("_data_dir", ""),
        }

    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Compute per-episode metrics from subtask completion data.

        Reads serialized subtask arrays from the last step's info dict.
        Returns metrics for result.json AND data for aggregate_results().
        """
        if not trajectory:
            return self._empty_metrics()

        last = trajectory[-1]
        info = last.info

        # Parse subtask arrays from the last step's info
        successes = json.loads(info.get("subtask_successes", "[]"))
        oracle_successes = json.loads(info.get("subtask_oracle_successes", "[]"))
        nav_errors = json.loads(info.get("subtask_nav_errors", "[]"))
        nav_steps = json.loads(info.get("subtask_nav_steps", "[]"))
        gt_steps = json.loads(info.get("gt_steps", "[]"))
        gt_paths = json.loads(info.get("gt_paths", "[]"))

        # Overall success: all subtasks succeeded
        task_success = 1.0 if successes and all(s == 1 for s in successes) else 0.0
        oracle_success = 1.0 if oracle_successes and all(s == 1 for s in oracle_successes) else 0.0

        # SPL: success * (gt_total / max(gt_total, actual_total))
        total_gt = sum(gt_steps) if gt_steps else 0
        total_actual = sum(nav_steps) if nav_steps else 0
        spl = task_success * (total_gt / max(total_gt, total_actual)) if total_actual > 0 else 0.0

        # Navigation error: avg geodesic distance at stop across subtasks
        ne = sum(nav_errors) / len(nav_errors) if nav_errors else 0.0

        return {
            "task_success": task_success,
            "oracle_success": oracle_success,
            "spl": spl,
            "navigation_error": ne,
            "num_steps": float(len(trajectory)),
            "num_subtasks": float(len(successes)),
            "subtasks_completed": float(sum(successes)),
            # Store raw arrays as JSON strings for aggregate_results
            "_subtask_successes": json.dumps(successes),
            "_subtask_oracle_successes": json.dumps(oracle_successes),
            "_subtask_nav_errors": json.dumps(nav_errors),
            "_subtask_nav_steps": json.dumps(nav_steps),
            "_gt_steps": json.dumps(gt_steps),
            "_gt_paths": json.dumps(gt_paths),
        }

    def aggregate_results(self, records: list[EpisodeRecord]) -> dict:
        """Compute all 8 LHPR-VLN metrics using vendored NavigationMetrics.

        Returns nested dict grouped by robot type:
        - base: all episodes
        - spot: Spot robot episodes only
        - stretch: Stretch robot episodes only

        Metric definitions (from CVPR-25 paper):
        - SR, OSR, SPL, NE, ISR, CSR, CGT, TAR, contest_score
        """
        if not records:
            return {}

        # Group records by robot type
        groups: dict[str, list[EpisodeRecord]] = {}
        for r in records:
            robot = r.episode.get("robot", "spot")
            groups.setdefault(robot, []).append(r)

        output: dict = {}

        # Base group: all episodes
        base = self._compute_group_metrics(records)
        base["num_episodes"] = len(records)
        base["success_rate"] = base["SR"]
        base.update(self._compute_step_stats(records))
        output["base"] = base

        # Per-robot groups
        for robot_type, group_records in sorted(groups.items()):
            group = self._compute_group_metrics(group_records)
            group["num_episodes"] = len(group_records)
            group.update(self._compute_step_stats(group_records))
            output[robot_type] = group

        return output

    def _compute_group_metrics(
        self, records: list[EpisodeRecord]
    ) -> dict[str, float]:
        """Compute LHPR-VLN metrics for a group of records."""
        metrics = NavigationMetrics()

        for r in records:
            er = r.episode_results
            successes = json.loads(er.get("_subtask_successes", "[]"))
            oracle_successes = json.loads(er.get("_subtask_oracle_successes", "[]"))
            nav_errors = json.loads(er.get("_subtask_nav_errors", "[]"))
            nav_steps = json.loads(er.get("_subtask_nav_steps", "[]"))
            gt_steps = json.loads(er.get("_gt_steps", "[]"))
            gt_paths = json.loads(er.get("_gt_paths", "[]"))

            success = 1 if successes and all(s == 1 for s in successes) else 0
            oracle_success = 1 if oracle_successes and all(s == 1 for s in oracle_successes) else 0
            total_gt = sum(gt_steps) if gt_steps else 0
            total_actual = sum(nav_steps) if nav_steps else 0
            avg_ne = sum(nav_errors) / len(nav_errors) if nav_errors else 0.0

            metrics.add_sample(
                success=success,
                gt_step=total_gt,
                path_step=total_actual,
                oracle_success=oracle_success,
                navigation_error=avg_ne,
                subtask_successes=successes,
                subtask_path_step=gt_steps,
                gt_length=gt_paths,
                error_length=nav_errors,
            )

        result = metrics.compute()

        tar = result.get("tar", 0)
        isr = result.get("independent_success_rate", 0)
        csr = result.get("conditional_success_rate", 0)
        cgt = result.get("conditional_path_length", 0)
        contest_score = 0.4 * tar + 0.2 * isr + 0.2 * csr + 0.2 * cgt

        return {
            "SR": round(result["success_rate"], 4),
            "OSR": round(result["oracle_success_rate"], 4),
            "SPL": round(result["spl"], 4),
            "NE": round(result["navigation_error"], 4),
            "ISR": round(result["independent_success_rate"], 4),
            "CSR": round(result["conditional_success_rate"], 4),
            "CGT": round(result["conditional_path_length"], 4),
            "TAR": round(result["tar"], 4),
            "contest_score": round(contest_score, 4),
        }

    @staticmethod
    def _compute_step_stats(records: list[EpisodeRecord]) -> dict[str, float]:
        """Compute step count statistics across episodes."""
        steps = [r.episode_results.get("num_steps", 0) for r in records]
        if not steps:
            return {"avg_steps": 0.0, "median_steps": 0.0, "max_steps_reached": 0}
        sorted_steps = sorted(steps)
        return {
            "avg_steps": round(sum(steps) / len(steps), 1),
            "median_steps": round(sorted_steps[len(sorted_steps) // 2], 1),
            "max_steps_reached": sum(1 for s in steps if s >= 500),
        }

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "task_success": 0.0,
            "oracle_success": 0.0,
            "spl": 0.0,
            "navigation_error": 0.0,
            "num_steps": 0.0,
            "num_subtasks": 0.0,
            "subtasks_completed": 0.0,
            "_subtask_successes": "[]",
            "_subtask_oracle_successes": "[]",
            "_subtask_nav_errors": "[]",
            "_subtask_nav_steps": "[]",
            "_gt_steps": "[]",
            "_gt_paths": "[]",
        }

    def _get_builtin_episodes(self) -> list[dict]:
        """Minimal episodes for testing without dataset download."""
        return [
            {
                "id": "test_0",
                "instruction": "Find the chair in the living room, then find the table in the kitchen.",
                "scene": "00384-ceJTwFNjqCt",
                "robot": "spot",
                "objects": ["chair", "table"],
                "regions": ["3", "5"],
                "rooms": ["living room", "kitchen"],
                "gt_steps": [40, 55],
                "subtask_list": ["Move_to('chair_3')", "Move_to('table_5')"],
                "num_targets": 2,
                "batch": "builtin",
            },
        ]
