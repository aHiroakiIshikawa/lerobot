#!/usr/bin/env python

"""
Evaluate a policy trained on pressure-extended dataset.

Usage:
python examples/custom_eval_with_pressure.py
"""

import logging
import time

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.pressure_sensor import PressureSensor
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Configuration
HF_MODEL_ID = "aHiroakiIshikawa/policy-with-pressure-pick-up-chocolate-act"
HF_DATASET_ID = "aHiroakiIshikawa/record-with-pressure-pick-up-chocolate"
FOLLOWER_PORT = "/dev/tty.usbmodem5AA90176051"
PRESSURE_SENSOR_PORT = "/dev/cu.usbmodem1101"
FOLLOWER_ID = "hishikawa_follower_arm"

# Single episode for 3 minutes
NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 180  # 3 minutes

# Enable rerun visualization
DISPLAY_DATA = True

# Pressure threshold in Newtons (raw value)
PRESSURE_THRESHOLD = 0.5  # 5N


def extend_state_with_pressure(obs_frame: dict, pressure: float) -> dict:
    """Add binary pressure value (0 or 1) to observation.state in the policy input format.
    
    Args:
        obs_frame: Observation frame dictionary
        pressure: Raw pressure value in Newtons
    
    Returns:
        Extended observation frame with binary pressure (0 or 1)
    """
    extended_frame = {}
    
    # Binarize pressure: 0 if below threshold, 1 if above
    binary_pressure = 1.0 if pressure >= PRESSURE_THRESHOLD else 0.0
    
    for key, value in obs_frame.items():
        if key == "observation.state":
            if isinstance(value, torch.Tensor):
                pressure_tensor = torch.tensor([[binary_pressure]], dtype=value.dtype, device=value.device)
                extended_frame[key] = torch.cat([value, pressure_tensor], dim=-1)
            else:
                extended_frame[key] = np.append(value, np.float32(binary_pressure))
        else:
            extended_frame[key] = value
    
    return extended_frame


def main():
    init_logging()
    
    device = torch.device("mps")
    
    camera_config = {
        "left": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "right": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    }
    
    robot_config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=camera_config,
        use_degrees=True,
    )
    
    robot = SO101Follower(robot_config)
    
    pressure_sensor = PressureSensor(
        port=PRESSURE_SENSOR_PORT,
        baudrate=9600
    )
    
    logging.info(f"Loading policy from {HF_MODEL_ID}")
    policy = ACTPolicy.from_pretrained(HF_MODEL_ID)
    policy.eval()
    policy.to(device)
    
    logging.info(f"Loading dataset metadata from {HF_DATASET_ID}")
    dataset_metadata = LeRobotDatasetMetadata(HF_DATASET_ID)
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        HF_MODEL_ID,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    robot.connect()
    
    pressure_sensor.connect(max_retries=3, retry_delay=1.0)
    logging.info("Pressure sensor connected successfully")
    
    listener, events = init_keyboard_listener()
    
    # Initialize rerun visualization
    if DISPLAY_DATA:
        init_rerun(session_name="pressure_evaluation")
    
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")
    
    log_say("Evaluation started - 3 minutes continuous", play_sounds=False)
    print("\n" + "="*50)
    print("3 MINUTE CONTINUOUS EVALUATION")
    print(f"  Pressure Threshold: {PRESSURE_THRESHOLD}N")
    print("  ESC: Stop evaluation")
    print("="*50 + "\n")
    
    # Reset policy state
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    
    start_episode_t = time.perf_counter()
    frame_count = 0
    
    # Single 3-minute episode loop
    while time.perf_counter() - start_episode_t < EPISODE_TIME_SEC:
        start_loop_t = time.perf_counter()
        
        if events["stop_recording"]:
            break
        
        obs = robot.get_observation()
        
        try:
            pressure = pressure_sensor.get_force()
        except Exception as e:
            logging.warning(f"Pressure read error: {e}, using 0.0")
            pressure = 0.0
        
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_metadata.features,
            device=device,
        )
        
        # Binary pressure will be computed inside this function
        obs_frame_with_pressure = extend_state_with_pressure(obs_frame, pressure)
        
        obs_dict = preprocessor(obs_frame_with_pressure)
        
        with torch.inference_mode():
            if policy.config.use_amp:
                with torch.autocast(device_type=device.type):
                    action_tensor = policy.select_action(obs_dict)
            else:
                action_tensor = policy.select_action(obs_dict)
        
        action_processed = postprocessor(action_tensor)
        robot_action = make_robot_action(action_processed, dataset_metadata.features)
        robot.send_action(robot_action)
        
        # Log data to rerun for visualization
        if DISPLAY_DATA:
            # Split data into observation and action dicts
            obs_data = {}
            action_data = {}
            
            # Add observations (without "observation." prefix for log_rerun_data)
            for key, value in obs.items():
                obs_data[key] = value
            
            # Log both raw and binary pressure for visualization
            binary_pressure = 1.0 if pressure >= PRESSURE_THRESHOLD else 0.0
            obs_data["pressure_raw"] = np.float32(pressure)
            obs_data["pressure_binary"] = np.float32(binary_pressure)
            
            # Add actions (without "action." prefix for log_rerun_data)
            for key, value in robot_action.items():
                action_data[key] = value
            
            log_rerun_data(observation=obs_data, action=action_data)
        
        frame_count += 1
        
        # Progress display every second
        elapsed = time.perf_counter() - start_episode_t
        remaining = EPISODE_TIME_SEC - elapsed
        if frame_count % 30 == 0:
            binary_state = "ON" if pressure >= PRESSURE_THRESHOLD else "OFF"
            print(f"  Time: {elapsed:.0f}s / {EPISODE_TIME_SEC}s | Remaining: {remaining:.0f}s | Frames: {frame_count} | Pressure: {pressure:.3f}N ({binary_state})", end="\r")
        
        dt_s = time.perf_counter() - start_loop_t
        sleep_time = 1 / FPS - dt_s
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    print()
    
    # Cleanup
    robot.disconnect()
    pressure_sensor.disconnect()
    
    if listener is not None:
        listener.stop()
    
    total_time = time.perf_counter() - start_episode_t
    log_say("Evaluation finished", play_sounds=False, blocking=True)
    print(f"\nCompleted: {total_time:.1f}s, {frame_count} frames")


if __name__ == "__main__":
    main()