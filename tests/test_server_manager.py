"""Tests for ServerManager — vLLM lifecycle management."""
import socket
import pytest


class TestServerManagerInit:
    def test_stores_config(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "Qwen/Qwen2.5-VL-72B", port=8080,
                           server_kwargs={"tensor_parallel_size": 4})
        assert sm.backend == "vllm"
        assert sm.model == "Qwen/Qwen2.5-VL-72B"
        assert sm.port == 8080
        assert sm.server_kwargs == {"tensor_parallel_size": 4}

    def test_default_port(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "some-model")
        assert sm.port == 8080


class TestBuildCommand:
    def test_vllm_basic(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "Qwen/Qwen2.5-VL-72B", port=9090)
        cmd, env = sm._build_command()
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model" in cmd
        assert "Qwen/Qwen2.5-VL-72B" in cmd
        assert "--port" in cmd
        assert "9090" in cmd

    def test_vllm_with_kwargs(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test-model", port=8080,
                           server_kwargs={"tensor_parallel_size": 4,
                                          "gpu_memory_utilization": 0.9})
        cmd, env = sm._build_command()
        assert "--tensor-parallel-size" in cmd
        assert "4" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd

    def test_unsupported_backend_raises(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("unknown_backend", "test-model")
        with pytest.raises(ValueError, match="Unsupported server backend"):
            sm._build_command()


class TestPortCheck:
    def test_port_available(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test", port=19876)
        sm._check_port()  # should not raise

    def test_port_taken(self):
        from easi.llm.server_manager import ServerManager
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 19877))
        sock.listen(1)
        try:
            sm = ServerManager("vllm", "test", port=19877)
            with pytest.raises(RuntimeError, match="already in use"):
                sm._check_port()
        finally:
            sock.close()


class TestContextManager:
    def test_context_manager_calls_stop(self):
        from unittest.mock import MagicMock
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test", port=8080)
        sm.start = MagicMock(return_value="http://localhost:8080/v1")
        sm.stop = MagicMock()
        with sm as url:
            assert url == "http://localhost:8080/v1"
        sm.stop.assert_called_once()


class TestIsRunning:
    def test_not_running_when_no_process(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test")
        assert sm.is_running() is False


class TestDefaultVllmFlags:
    def test_server_manager_default_vllm_flags(self):
        """ServerManager should include default vLLM flags (prefix caching, quiet logs)."""
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(backend="vllm", model="test-model", port=9999)
        cmd, _ = mgr._build_command()
        assert "--enable-prefix-caching" in cmd
        assert "--disable-log-requests" in cmd

    def test_server_manager_user_can_override_defaults(self):
        """User-provided server_kwargs should override defaults."""
        from easi.llm.server_manager import ServerManager
        mgr = ServerManager(
            backend="vllm", model="test-model", port=9999,
            server_kwargs={"enable_prefix_caching": False},
        )
        cmd, _ = mgr._build_command()
        # False bool flags should NOT appear in command
        assert "--enable-prefix-caching" not in cmd


class TestCudaVisibleDevices:
    def test_server_manager_sets_cuda_visible_devices(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test-model", cuda_visible_devices="0,1")
        _cmd, env = sm._build_command()
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_server_manager_no_cuda_by_default(self):
        from easi.llm.server_manager import ServerManager
        sm = ServerManager("vllm", "test-model")
        _cmd, env = sm._build_command()
        assert "CUDA_VISIBLE_DEVICES" not in env


class TestMultiServerManager:
    def test_multi_server_manager_starts_n_instances(self):
        """MultiServerManager should start N ServerManager instances with correct GPU assignment."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM:
            mock_instance = MagicMock()
            mock_instance.start.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1],
                base_port=8000,
            )
            urls = mgr.start()

            assert len(urls) == 2
            assert MockSM.call_count == 2

            # Check GPU assignment: instance 0 → GPU 0, instance 1 → GPU 1
            calls = MockSM.call_args_list
            assert calls[0].kwargs["cuda_visible_devices"] == "0"
            assert calls[0].kwargs["port"] == 8000
            assert calls[1].kwargs["cuda_visible_devices"] == "1"
            assert calls[1].kwargs["port"] == 8001

    def test_multi_server_manager_tp2_gpu_assignment(self):
        """With TP=2, each instance should get 2 GPUs."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM:
            mock_instance = MagicMock()
            mock_instance.start.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(
                model="test-model",
                num_instances=2,
                gpu_ids=[0, 1, 2, 3],
                base_port=8000,
            )
            urls = mgr.start()

            calls = MockSM.call_args_list
            assert calls[0].kwargs["cuda_visible_devices"] == "0,1"
            assert calls[1].kwargs["cuda_visible_devices"] == "2,3"

    def test_multi_server_manager_stop_all(self):
        """stop() should stop all managed instances."""
        from unittest.mock import patch, MagicMock
        from easi.llm.server_manager import MultiServerManager

        with patch("easi.llm.server_manager.ServerManager") as MockSM:
            mock_instance = MagicMock()
            mock_instance.start.return_value = "http://localhost:8000/v1"
            MockSM.return_value = mock_instance

            mgr = MultiServerManager(model="m", num_instances=2, gpu_ids=[0, 1])
            mgr.start()
            mgr.stop()

            assert mock_instance.stop.call_count == 2

    def test_multi_server_manager_validates_gpu_count(self):
        """Should raise if GPUs don't divide evenly across instances."""
        from easi.llm.server_manager import MultiServerManager

        with pytest.raises(ValueError, match="divide.*evenly"):
            MultiServerManager(model="m", num_instances=3, gpu_ids=[0, 1])
