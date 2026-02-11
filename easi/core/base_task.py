"""Abstract base class for tasks (benchmarks).

A task owns:
- Which simulator+version to use (pinned via task.yaml)
- The action space available to the agent
- The dataset (episodes to evaluate on)
- Episode-to-simulator mapping (format_reset_config)
- Success criteria and metrics (evaluate_episode)

Concrete tasks subclass this and implement:
- get_task_yaml_path() — where the task.yaml lives
- format_reset_config() — adapter from dataset episodes to simulator configs
- evaluate_episode() — compute metrics for a completed episode
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from easi.core.episode import StepResult
from easi.core.exceptions import DatasetError

logger = logging.getLogger("easi.core.base_task")


class BaseTask(ABC):
    """Abstract base for all tasks (benchmarks)."""

    def __init__(self, data_dir: Path | None = None):
        self._config = self._load_config()
        self._episodes: list[dict] | None = None
        self._data_dir = data_dir

    @abstractmethod
    def get_task_yaml_path(self) -> Path:
        """Return path to this task's task.yaml."""
        ...

    @abstractmethod
    def format_reset_config(self, episode: dict) -> dict:
        """Translate a dataset episode into simulator reset kwargs.

        This is the core adapter method between dataset format and simulator API.
        """
        ...

    @abstractmethod
    def evaluate_episode(
        self, episode: dict, trajectory: list[StepResult]
    ) -> dict[str, float]:
        """Evaluate a completed episode, returning a metric dict.

        Example return: {"success": 1.0, "spl": 0.73, "distance_to_goal": 0.15}
        """
        ...

    # --- Shared implementation ---

    @property
    def name(self) -> str:
        return self._config["name"]

    @property
    def simulator_key(self) -> str:
        """Returns e.g. 'dummy:v1' — used to look up from simulator registry."""
        return self._config["simulator"]

    @property
    def action_space(self) -> list[str]:
        return self._config["action_space"]

    @property
    def max_steps(self) -> int:
        return self._config.get("max_steps", 500)

    def download_dataset(self) -> Path:
        """Download dataset if needed. Returns path to local data directory.

        - source=local: validate path exists, return it
        - source=huggingface: download via huggingface_hub, cache locally
        """
        dataset_config = self._config.get("dataset", {})
        source = dataset_config.get("source", "local")

        if source == "local":
            path = dataset_config.get("path")
            if path is None:
                # Use built-in episodes (no download needed)
                return Path()
            local_path = Path(path)
            if not local_path.exists():
                raise DatasetError(f"Local dataset path does not exist: {local_path}")
            return local_path

        elif source == "huggingface":
            return self._download_huggingface(dataset_config)

        else:
            raise DatasetError(f"Unknown dataset source: {source}")

    def load_episodes(self) -> list[dict]:
        """Load and return all episodes from the dataset."""
        if self._episodes is not None:
            return self._episodes

        self._episodes = self._load_episodes_from_config()
        logger.info("Loaded %d episodes for task %s", len(self._episodes), self.name)
        return self._episodes

    def get_episode(self, index: int) -> dict:
        """Get a single episode by index."""
        episodes = self.load_episodes()
        if index < 0 or index >= len(episodes):
            raise IndexError(f"Episode index {index} out of range [0, {len(episodes)})")
        return episodes[index]

    def __len__(self) -> int:
        """Number of episodes."""
        return len(self.load_episodes())

    def _load_config(self) -> dict:
        """Load task.yaml."""
        yaml_path = self.get_task_yaml_path()
        if not yaml_path.exists():
            raise DatasetError(f"Task config not found: {yaml_path}")
        return yaml.safe_load(yaml_path.read_text())

    def _load_episodes_from_config(self) -> list[dict]:
        """Load episodes from the dataset directory.

        Subclasses can override this for custom episode loading logic.
        Default implementation looks for episodes.json in the data directory.
        """
        data_dir = self._data_dir or self.download_dataset()
        if not data_dir or data_dir == Path():
            # No data dir — subclass should provide built-in episodes
            return self._get_builtin_episodes()

        episodes_file = data_dir / "episodes.json"
        if episodes_file.exists():
            return json.loads(episodes_file.read_text())

        raise DatasetError(
            f"No episodes.json found in {data_dir}. "
            f"Override _load_episodes_from_config() for custom loading."
        )

    def _get_builtin_episodes(self) -> list[dict]:
        """Return built-in episodes when no dataset download is needed.

        Override in subclasses that provide built-in test episodes (e.g., DummyTask).
        """
        return []

    def _download_huggingface(self, config: dict) -> Path:
        """Download a dataset from HuggingFace Hub with file-based locking."""
        from easi.utils.locking import file_lock
        from easi.utils.paths import get_datasets_dir, get_locks_dir

        repo_id = config["repo_id"]
        lock_path = get_locks_dir() / f"dataset_{repo_id.replace('/', '_')}.lock"

        with file_lock(lock_path):
            target = get_datasets_dir() / repo_id.replace("/", "_")
            if target.exists():
                logger.info("Dataset %s already cached at %s", repo_id, target)
                return target

            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    repo_type="dataset",
                )
                logger.info("Downloaded dataset %s to %s", repo_id, target)
                return target
            except ImportError:
                raise DatasetError(
                    "huggingface_hub is required for HuggingFace downloads. "
                    "Install with: pip install huggingface_hub"
                )
            except Exception as e:
                raise DatasetError(f"Failed to download {repo_id}: {e}")
