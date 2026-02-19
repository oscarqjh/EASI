"""Tests for ParallelRunner — parallel episode evaluation."""
import json
import pytest


class TestParallelRunnerInit:
    """Test ParallelRunner construction and validation."""

    def test_inherits_evaluation_runner(self):
        from easi.evaluation.parallel_runner import ParallelRunner
        from easi.evaluation.runner import EvaluationRunner
        assert issubclass(ParallelRunner, EvaluationRunner)

    def test_num_parallel_stored(self):
        from easi.evaluation.parallel_runner import ParallelRunner
        runner = ParallelRunner(
            task_name="dummy_task", num_parallel=4,
            agent_type="dummy",
        )
        assert runner.num_parallel == 4

    def test_defaults_to_2_workers(self):
        from easi.evaluation.parallel_runner import ParallelRunner
        runner = ParallelRunner(
            task_name="dummy_task",
            agent_type="dummy",
        )
        assert runner.num_parallel == 2


class TestParallelRunnerValidation:
    """Test that unsupported configs are rejected."""

    def test_resume_raises_not_implemented(self, tmp_path):
        from easi.evaluation.parallel_runner import ParallelRunner
        resume_dir = tmp_path / "fake_run"
        resume_dir.mkdir()
        (resume_dir / "config.json").write_text('{"cli_options": {}}')

        runner = ParallelRunner(
            task_name="dummy_task", num_parallel=2,
            agent_type="dummy", resume_dir=str(resume_dir),
        )
        with pytest.raises(NotImplementedError, match="does not support --resume"):
            runner.run()

    def test_local_vllm_raises_not_implemented(self, tmp_path):
        from easi.evaluation.parallel_runner import ParallelRunner
        runner = ParallelRunner(
            task_name="dummy_task", num_parallel=2,
            agent_type="react", backend="vllm",
            model="some-model", output_dir=str(tmp_path),
        )
        with pytest.raises(NotImplementedError, match="does not support local vLLM"):
            runner.run()


class TestParallelRunnerWithDummy:
    """Integration test: parallel run with dummy agent + dummy simulator."""

    def test_parallel_dummy_produces_results(self, tmp_path):
        """Run dummy_task with 2 workers, verify all episodes complete."""
        from easi.evaluation.parallel_runner import ParallelRunner

        runner = ParallelRunner(
            task_name="dummy_task",
            num_parallel=2,
            agent_type="dummy",
            output_dir=str(tmp_path),
            max_episodes=3,
        )
        results = runner.run()

        # All 3 episodes should complete
        assert len(results) == 3

        # Results should be in episode order (sorted by index)
        episode_ids = [r["episode_id"] for r in results]
        # dummy_task episodes have sequential ids
        assert len(set(episode_ids)) == 3  # all unique

        # Summary should exist with parallel-specific fields
        run_dirs = list((tmp_path / "dummy_task").iterdir())
        assert len(run_dirs) == 1
        summary_path = run_dirs[0] / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["num_parallel"] == 2
        assert "wall_clock_seconds" in summary

    def test_parallel_single_episode(self, tmp_path):
        """With 1 episode and 2 workers, only 1 worker should do work."""
        from easi.evaluation.parallel_runner import ParallelRunner

        runner = ParallelRunner(
            task_name="dummy_task",
            num_parallel=2,
            agent_type="dummy",
            output_dir=str(tmp_path),
            max_episodes=1,
        )
        results = runner.run()
        assert len(results) == 1

    def test_config_json_records_num_parallel(self, tmp_path):
        """config.json should include num_parallel."""
        from easi.evaluation.parallel_runner import ParallelRunner

        runner = ParallelRunner(
            task_name="dummy_task",
            num_parallel=3,
            agent_type="dummy",
            output_dir=str(tmp_path),
            max_episodes=1,
        )
        runner.run()

        run_dirs = list((tmp_path / "dummy_task").iterdir())
        config = json.loads((run_dirs[0] / "config.json").read_text())
        assert config["num_parallel"] == 3

    def test_episode_dirs_created(self, tmp_path):
        """Each episode should get its own directory with result.json."""
        from easi.evaluation.parallel_runner import ParallelRunner

        runner = ParallelRunner(
            task_name="dummy_task",
            num_parallel=2,
            agent_type="dummy",
            output_dir=str(tmp_path),
            max_episodes=2,
        )
        runner.run()

        run_dirs = list((tmp_path / "dummy_task").iterdir())
        episodes_dir = run_dirs[0] / "episodes"
        episode_dirs = sorted(episodes_dir.iterdir())
        assert len(episode_dirs) == 2
        for ed in episode_dirs:
            assert (ed / "result.json").exists()


class TestCLIParallelArg:
    """Test --num-parallel CLI argument parsing."""

    def test_num_parallel_parsed(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task", "--num-parallel", "4"])
        assert args.num_parallel == 4

    def test_num_parallel_default_is_none(self):
        from easi.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["start", "dummy_task"])
        assert args.num_parallel is None
