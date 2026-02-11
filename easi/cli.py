"""EASI CLI entry point.

Usage:
    easi env list|install|check <simulator>
    easi task list|info|download <task>
    easi sim test <simulator>
    easi llm-server [--port PORT] [--mode MODE]
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easi",
        description="EASI - Embodied Reasoning Evaluation for Spatial Intelligence",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable INFO logging"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    subparsers = parser.add_subparsers(dest="command")

    # --- env command group ---
    env_parser = subparsers.add_parser("env", help="Manage simulator environments")
    env_sub = env_parser.add_subparsers(dest="env_action")

    env_sub.add_parser("list", help="List available simulators and versions")

    env_install = env_sub.add_parser("install", help="Install a simulator environment")
    env_install.add_argument("simulator", type=str, help="e.g., 'dummy' or 'ai2thor:v2_1_0'")

    env_check = env_sub.add_parser("check", help="Check if environment is ready")
    env_check.add_argument("simulator", type=str)

    # --- task command group ---
    task_parser = subparsers.add_parser("task", help="Manage tasks (benchmarks)")
    task_sub = task_parser.add_subparsers(dest="task_action")

    task_sub.add_parser("list", help="List available tasks")

    task_info = task_sub.add_parser("info", help="Show task details")
    task_info.add_argument("task", type=str, help="e.g., 'dummy_task'")

    task_download = task_sub.add_parser("download", help="Download task dataset")
    task_download.add_argument("task", type=str)

    # --- sim command group ---
    sim_parser = subparsers.add_parser("sim", help="Control simulators")
    sim_sub = sim_parser.add_subparsers(dest="sim_action")

    sim_test = sim_sub.add_parser("test", help="Run a smoke test (reset + N steps)")
    sim_test.add_argument("simulator", type=str, help="e.g., 'dummy' or 'ai2thor:v5_0_0'")
    sim_test.add_argument("--steps", type=int, default=5, help="Number of steps")

    # --- llm-server command ---
    llm_parser = subparsers.add_parser("llm-server", help="Start dummy LLM server")
    llm_parser.add_argument("--port", type=int, default=8000)
    llm_parser.add_argument("--host", type=str, default="127.0.0.1")
    llm_parser.add_argument("--mode", choices=["fixed", "random"], default="random")
    llm_parser.add_argument(
        "--action-space", type=str, nargs="+",
        default=["MoveAhead", "TurnLeft", "TurnRight", "Stop"],
    )

    return parser


# --- Command handlers ---

def cmd_env_list() -> None:
    from easi.simulators.registry import get_simulator_entry, list_simulators

    sims = list_simulators()
    if not sims:
        print("No simulators found.")
        return

    # Deduplicate: show each name:version pair once
    seen = set()
    for key in sims:
        entry = get_simulator_entry(key)
        pair = f"{entry.name}:{entry.version}"
        if pair in seen:
            continue
        seen.add(pair)
        default_marker = " (default)" if key == entry.name else ""
        print(f"  {pair}{default_marker}  — {entry.description}")


def cmd_env_install(simulator: str) -> None:
    from easi.simulators.registry import load_env_manager_class

    EnvManagerClass = load_env_manager_class(simulator)
    env_manager = EnvManagerClass()
    print(f"Installing environment: {env_manager.get_env_name()}")
    env_manager.install()
    print("Done.")


def cmd_env_check(simulator: str) -> None:
    from easi.simulators.registry import load_env_manager_class

    EnvManagerClass = load_env_manager_class(simulator)
    env_manager = EnvManagerClass()

    missing = env_manager.check_system_deps()
    if missing:
        print(f"Missing system deps: {missing}")

    if env_manager.env_is_ready():
        print(f"Environment {env_manager.get_env_name()} is ready.")
        print(f"Python: {env_manager.get_python_executable()}")
    else:
        print(f"Environment {env_manager.get_env_name()} is NOT ready.")
        print("Run: easi env install " + simulator)


def cmd_task_list() -> None:
    from easi.tasks.registry import get_task_entry, list_tasks

    tasks = list_tasks()
    if not tasks:
        print("No tasks found.")
        return

    for name in tasks:
        entry = get_task_entry(name)
        print(f"  {name}  — {entry.display_name} (simulator: {entry.simulator_key})")


def cmd_task_info(task_name: str) -> None:
    from easi.tasks.registry import get_task_entry

    entry = get_task_entry(task_name)
    print(f"Task: {entry.display_name}")
    print(f"  Name:        {entry.name}")
    print(f"  Description: {entry.description}")
    print(f"  Simulator:   {entry.simulator_key}")
    print(f"  Actions:     {', '.join(entry.action_space)}")
    print(f"  Max steps:   {entry.max_steps}")


def cmd_task_download(task_name: str) -> None:
    from easi.tasks.registry import load_task_class

    TaskClass = load_task_class(task_name)
    task = TaskClass()
    path = task.download_dataset()
    if path and str(path):
        print(f"Dataset ready at: {path}")
    else:
        print("Task uses built-in episodes (no download needed).")


def cmd_sim_test(simulator: str, steps: int) -> None:
    from easi.core.episode import Action
    from easi.simulators.registry import load_env_manager_class, load_simulator_class
    from easi.simulators.subprocess_runner import SubprocessRunner

    EnvManagerClass = load_env_manager_class(simulator)
    SimClass = load_simulator_class(simulator)

    env_manager = EnvManagerClass()
    sim = SimClass()

    print(f"Testing {simulator}...")
    print(f"  Python: {env_manager.get_python_executable()}")

    runner = SubprocessRunner(
        python_executable=env_manager.get_python_executable(),
        bridge_script_path=sim._get_bridge_script_path(),
        needs_display=env_manager.needs_display,
        xvfb_screen_config=env_manager.xvfb_screen_config,
    )

    try:
        runner.launch()
        sim.set_runner(runner)

        print("  Reset... ", end="", flush=True)
        obs = sim.reset("smoke_test_001")
        print(f"OK (rgb: {obs.rgb_path})")

        for i in range(steps):
            action = Action(action_name="MoveAhead")
            result = sim.step(action)
            print(f"  Step {i+1}: done={result.done}, reward={result.reward}")
            if result.done:
                break

        print("  Closing... ", end="", flush=True)
        sim.close()
        print("OK")
        print("Smoke test passed!")

    except Exception as e:
        print(f"\nSmoke test FAILED: {e}")
        sim.close()
        sys.exit(1)


def cmd_llm_server(host: str, port: int, mode: str, action_space: list[str]) -> None:
    from easi.llm.dummy_server import run_server

    run_server(host=host, port=port, mode=mode, action_space=action_space)


# --- Main ---

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Set up logging
    if args.debug:
        from easi.utils.logging import setup_logging
        setup_logging("DEBUG")
    elif args.verbose:
        from easi.utils.logging import setup_logging
        setup_logging("INFO")

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Dispatch commands
    if args.command == "env":
        if args.env_action == "list":
            cmd_env_list()
        elif args.env_action == "install":
            cmd_env_install(args.simulator)
        elif args.env_action == "check":
            cmd_env_check(args.simulator)
        else:
            parser.parse_args(["env", "--help"])

    elif args.command == "task":
        if args.task_action == "list":
            cmd_task_list()
        elif args.task_action == "info":
            cmd_task_info(args.task)
        elif args.task_action == "download":
            cmd_task_download(args.task)
        else:
            parser.parse_args(["task", "--help"])

    elif args.command == "sim":
        if args.sim_action == "test":
            cmd_sim_test(args.simulator, args.steps)
        else:
            parser.parse_args(["sim", "--help"])

    elif args.command == "llm-server":
        cmd_llm_server(args.host, args.port, args.mode, args.action_space)


if __name__ == "__main__":
    main()
