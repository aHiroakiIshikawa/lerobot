#!/usr/bin/env python

"""
Records a dataset with pressure binary values in observation.environment_state.

The pressure value is binarized using a threshold (PRESSURE_THRESHOLD = 0.2):
- pressure >= 0.2 -> 1.0 (grasping)
- pressure < 0.2  -> 0.0 (not grasping)

This allows training with or without pressure information:
- Without pressure: Use default training config (observation.environment_state is ignored)
- With pressure: Add observation.environment_state to input_features in training config

Usage:
rm -rf '/Users/hiroaki.ishikawa/.cache/huggingface/lerobot/aHiroakiIshikawa/record-with-pressure-green-cylinder'
python examples/custom_record_with_pressure_binary.py \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5AA90176051 \
        --robot.id=hishikawa_follower_arm \
        --robot.cameras="{ left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
        --teleop.type=so101_leader \
        --teleop.port=/dev/tty.usbmodem5A7A0161151 \
        --teleop.id=hishikawa_leader_arm \
        --teleop.pressure_sensor_enabled=true \
        --teleop.pressure_sensor_port=/dev/cu.usbmodem1101 \
        --display_data=true \
        --dataset.repo_id=${HF_USER}/record-with-pressure-green-cylinder \
        --dataset.episode_time_s=30 \
        --dataset.reset_time_s=5 \
        --dataset.num_episodes=50 \
        --dataset.single_task="Put the green cylinder into the white cup" \
        --dataset.push_to_hub=true \
        --dataset.private=true
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun


# 圧力の閾値
PRESSURE_THRESHOLD = 0.2


@dataclass
class PressureRecordConfig(RecordConfig):
    """Configuration for recording with pressure sensor."""
    # Inherits all fields from RecordConfig (robot, teleop, dataset, etc.)
    pass


def create_pressure_extended_features(features: dict, pressure_dim: int = 1) -> dict:
    """Create features dict with pressure binary state added to observation.environment_state."""
    extended_features = features.copy()
    
    # observation.environment_stateを追加（バイナリ値: 0.0 or 1.0）
    extended_features[OBS_ENV_STATE] = {
        "dtype": "float32",
        "shape": [pressure_dim],
        "names": ["pressure.binary"],
    }
    
    logging.info(f"Added {OBS_ENV_STATE}: shape [{pressure_dim}], threshold={PRESSURE_THRESHOLD}")
    
    return extended_features


def log_pressure_data(obs: RobotObservation, action: RobotAction, pressure: float, pressure_binary: float, frame_idx: int):
    """Log pressure data to Rerun."""
    rr.set_time_sequence("frame", frame_idx)
    
    # Log camera images
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            rr.log(f"images/{key}", rr.Image(value))
    
    # 全てのジョイント位置を同じ親パス "joints" 配下にログ
    for key, value in obs.items():
        if ".pos" in key:
            name = key.replace(".pos", "")
            rr.log(f"joints/obs_{name}", rr.Scalars(float(value)))
    
    for key, value in action.items():
        if ".pos" in key:
            name = key.replace(".pos", "")
            rr.log(f"joints/act_{name}", rr.Scalars(float(value)))
    
    # 圧力（生値とバイナリ値）をログ
    rr.log("pressure/raw", rr.Scalars(float(pressure)))
    rr.log("pressure/binary", rr.Scalars(float(pressure_binary)))


@safe_stop_image_writer
def record_loop_with_pressure(
    robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset: LeRobotDataset | None = None,
    teleop=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    has_pressure_sensor: bool = False,
):
    """Record loop that saves to dataset with pressure.
    
    This function follows the same structure as lerobot_record.py's record_loop():
    1. Get robot observation
    2. Process observation
    3. Get teleop action
    4. Process action
    5. Send action to robot
    6. Save to dataset
    7. Log to visualization
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")
    
    timestamp = 0
    start_episode_t = time.perf_counter()
    frame_count = 0
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Process observation (default is IdentityProcessor)
        obs_processed = robot_observation_processor(obs)

        # Get action from teleoperator
        if teleop is not None:
            act = teleop.get_action()
            
            # Process teleop action (default is IdentityProcessor)
            act_processed = teleop_action_processor((act, obs))
            
            # Process robot action (default is IdentityProcessor)
            robot_action_to_send = robot_action_processor((act_processed, obs))
            
            # Send action to robot
            robot.send_action(robot_action_to_send)
        else:
            logging.warning("No teleoperator provided, skipping action")
            continue

        # Get pressure value
        if has_pressure_sensor:
            pressure = teleop.read_pressure_sensor()
        else:
            pressure = 0.0

        # 圧力を閾値でバイナリ化
        pressure_binary = 1.0 if pressure >= PRESSURE_THRESHOLD else 0.0

        # Write to dataset (following the same pattern as lerobot_record.py)
        if dataset is not None:
            # Build observation frame (excluding environment_state which we add manually)
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
            
            # バイナリ化した圧力をenvironment_stateに追加
            observation_frame[OBS_ENV_STATE] = np.array([pressure_binary], dtype=np.float32)
            
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        # Log to visualization
        if display_data:
            try:
                log_pressure_data(obs_processed, act_processed, pressure, pressure_binary, frame_count)
            except Exception as e:
                logging.debug(f"Visualization error: {e}")

        frame_count += 1

        # Progress display
        if frame_count % 30 == 0:
            elapsed = time.perf_counter() - start_episode_t
            print(f"  Frames: {frame_count}, Time: {elapsed:.1f}s, Pressure: {pressure:.3f}N (binary: {pressure_binary})", end="\r")

        timestamp = time.perf_counter() - start_episode_t
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)
    
    print()  # New line after progress


@parser.wrap()
def record(cfg: PressureRecordConfig) -> LeRobotDataset:
    """Record dataset with pressure information.
    
    This function follows the same structure as lerobot_record.py's record():
    1. Initialize logging and visualization
    2. Create robot and teleoperator
    3. Create processors
    4. Create/load dataset with pressure features
    5. Connect hardware
    6. Recording loop with episode management
    7. Disconnect and push to hub
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        init_rerun(session_name="pressure_recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Create base features (same as lerobot_record.py)
    base_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    # Create pressure-extended features
    features = create_pressure_extended_features(base_features, pressure_dim=1)

    # Create or load dataset (following lerobot_record.py pattern)
    num_cameras = len(robot.cameras) if hasattr(robot, 'cameras') and robot.cameras else 0
    
    dataset = None
    listener = None
    
    try:
        if cfg.resume:
            # Resume existing dataset
            logging.info("Resuming existing dataset...")
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )
            
            # Start image writers if using cameras
            if num_cameras > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                )
            
            # Sanity check
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, features)
            
            logging.info(f"Resumed dataset with {dataset.meta.total_episodes} episodes")
        else:
            # Create new dataset
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        # Check for pressure sensor
        has_pressure_sensor = False
        if teleop is not None and hasattr(teleop, 'read_pressure_sensor'):
            has_pressure_sensor = True
            logging.info("Pressure sensor detected on teleoperator")
        else:
            logging.warning("No pressure sensor found. Pressure values will be 0.")

        listener, events = init_keyboard_listener()

        log_say("Recording started", cfg.play_sounds)
        print("\n" + "="*50)
        print("RECORDING CONTROLS:")
        print("  → (Right Arrow): End episode early and SAVE")
        print("  ← (Left Arrow):  DISCARD and re-record current episode")
        print("  ESC:             Stop recording completely")
        print("="*50 + "\n")

        # Recording loop with VideoEncodingManager (same as lerobot_record.py)
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                print(f"\n{'='*50}")
                print(f"--- Episode {dataset.num_episodes} ---")

                # Set episode timeline for Rerun
                if cfg.display_data:
                    rr.set_time_sequence("episode", dataset.num_episodes)

                # Main recording loop
                record_loop_with_pressure(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    has_pressure_sensor=has_pressure_sensor,
                )

                # Reset environment (same logic as lerobot_record.py)
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    
                    record_loop_with_pressure(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        has_pressure_sensor=has_pressure_sensor,
                    )

                # Handle re-recording (same logic as lerobot_record.py)
                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                # Save episode (same as lerobot_record.py)
                dataset.save_episode()
                recorded_episodes += 1
    finally:
        # Cleanup (same as lerobot_record.py)
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        # Push to hub (same as lerobot_record.py)
        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
