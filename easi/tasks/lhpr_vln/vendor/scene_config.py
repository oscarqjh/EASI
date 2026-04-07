"""Habitat-Sim configuration for LHPR-VLN scenes.

Vendored from LH-VLN/habitat_base/config.py with parameterization.
"""
from __future__ import annotations

import math


def _resolve_sensor_flags(sensors: dict | None) -> dict[str, bool]:
    """Convert a human-friendly sensors dict to flat setting-key booleans.

    Parameters
    ----------
    sensors : dict or None
        When *None* every sensor is enabled (backward-compatible default).
        Otherwise a dict with optional keys:
        - ``"rgb"``: list of directions, subset of ``["front", "left", "right"]``
        - ``"depth"``: list of directions, subset of ``["front", "left", "right"]``
        - ``"semantic"``: bool
        - ``"third_person"``: bool
    """
    _dir_map = {"front": "f", "left": "l", "right": "r"}

    if sensors is None:
        return {
            "color_sensor_f": True, "color_sensor_l": True, "color_sensor_r": True,
            "color_sensor_3rd": True,
            "depth_sensor_f": True, "depth_sensor_l": True, "depth_sensor_r": True,
            "semantic_sensor": True,
        }

    flags: dict[str, bool] = {
        "color_sensor_f": False, "color_sensor_l": False, "color_sensor_r": False,
        "color_sensor_3rd": False,
        "depth_sensor_f": False, "depth_sensor_l": False, "depth_sensor_r": False,
        "semantic_sensor": False,
    }

    for direction in sensors.get("rgb", []):
        flags[f"color_sensor_{_dir_map[direction]}"] = True
    for direction in sensors.get("depth", []):
        flags[f"depth_sensor_{_dir_map[direction]}"] = True
    if sensors.get("semantic", False):
        flags["semantic_sensor"] = True
    if sensors.get("third_person", False):
        flags["color_sensor_3rd"] = True

    return flags


def make_setting(scene_path: str, scene_dataset_path: str, robot: str,
                 width: int = 512, height: int = 512,
                 sensors: dict | None = None) -> dict:
    """Build Habitat-Sim settings dict for an HM3D scene."""
    sensor_height = 0.5 if robot == "spot" else 1.0
    settings = {
        "width": width,
        "height": height,
        "scene": scene_path,
        "scene_dataset": scene_dataset_path,
        "default_agent": 0,
        "sensor_height": sensor_height,
        "seed": 1,
        "enable_physics": False,
    }
    settings.update(_resolve_sensor_flags(sensors))
    return settings


def make_cfg(settings: dict, gpu_device_id: int = -1,
             forward_step_size: float = 0.25,
             turn_angle: float = 30.0):
    """Build Habitat-Sim Configuration from settings dict."""
    import habitat_sim
    import magnum as mn

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = gpu_device_id
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    if "scene_dataset" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]

    sensors = {
        "color_sensor_f": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
        "color_sensor_l": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "color_sensor_r": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "color_sensor_3rd": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"] + 0.5, 1.0),
            "orientation": [-math.pi / 4, 0.0, 0.0],
        },
        "depth_sensor_l": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "depth_sensor_f": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_sensor_r": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
    }
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings.get(sensor_uuid, False):
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            if sensor_uuid == "color_sensor_3rd":
                sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=forward_step_size)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_angle)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_angle)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
