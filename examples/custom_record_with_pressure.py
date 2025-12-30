#!/usr/bin/env python

"""
Records two datasets simultaneously:
1. Standard dataset (normal lerobot-record output)
2. Extended dataset with pressure values in observation.state

Usage:
rm -rf '/Users/hiroaki.ishikawa/.cache/huggingface/lerobot/aHiroakiIshikawa/record-normal-green-cylinder'
rm -rf '/Users/hiroaki.ishikawa/.cache/huggingface/lerobot/aHiroakiIshikawa/record-with-pressure-green-cylinder'
python examples/custom_record_with_pressure.py \
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
        --dataset.repo_id=${HF_USER}/record-normal-green-cylinder \
        --pressure_dataset.repo_id=${HF_USER}/record-with-pressure-green-cylinder \
        --dataset.episode_time_s=30 \
        --dataset.reset_time_s=5 \
        --dataset.num_episodes=50 \
        --dataset.single_task="Put the green cylinder into the white cup" \
        --dataset.push_to_hub=true \
        --dataset.private=true
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
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


@dataclass
class PressureDatasetConfig:
    """Config for the pressure-extended dataset."""
    repo_id: str = "user/record-with-pressure"
    root: str | Path | None = None


@dataclass
class DualRecordConfig(RecordConfig):
    """Extended config for recording two datasets."""
    pressure_dataset: PressureDatasetConfig = field(default_factory=PressureDatasetConfig)


def create_pressure_extended_features(features: dict, pressure_dim: int = 1) -> dict:
    """Create features dict with pressure added to observation.state."""
    extended_features = features.copy()
    
    if OBS_STATE in extended_features:
        obs_state = extended_features[OBS_STATE].copy()
        original_shape = obs_state.get("shape", [6])
        original_names = obs_state.get("names", [])
        
        # shapeを更新
        new_shape = [original_shape[0] + pressure_dim]
        obs_state["shape"] = new_shape
        
        # namesも更新（圧力の名前を追加）
        if original_names:
            new_names = list(original_names) + ["pressure.force"]
            obs_state["names"] = new_names
        
        extended_features[OBS_STATE] = obs_state
        
        logging.info(f"Extended observation.state: shape {original_shape} -> {new_shape}")
        logging.info(f"Extended observation.state names: {original_names} -> {obs_state.get('names', [])}")
    
    return extended_features


def log_pressure_data(obs: RobotObservation, action: RobotAction, pressure: float, frame_idx: int):
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
    
    # 圧力も同じグラフに追加
    rr.log("joints/pressure", rr.Scalars(float(pressure)))


@safe_stop_image_writer
def record_loop_with_pressure(
    robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    normal_dataset: LeRobotDataset | None = None,
    pressure_dataset: LeRobotDataset | None = None,
    teleop=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    has_pressure_sensor: bool = False,
):
    """Record loop that saves to both normal and pressure datasets.
    
    This function follows the same structure as lerobot_record.py's record_loop():
    1. Get robot observation
    2. Process observation
    3. Get teleop action
    4. Process action
    5. Send action to robot
    6. Save to dataset(s)
    7. Log to visualization
    """
    if normal_dataset is not None and normal_dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({normal_dataset.fps} != {fps}).")
    
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

        # Build observation frame for datasets
        if normal_dataset is not None and pressure_dataset is not None:
            observation_frame = build_dataset_frame(normal_dataset.features, obs_processed, prefix=OBS_STR)

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

        # Write to datasets (following the same pattern as lerobot_record.py)
        if normal_dataset is not None and pressure_dataset is not None:
            action_frame = build_dataset_frame(normal_dataset.features, act_processed, prefix=ACTION)
            
            # Normal dataset frame
            normal_frame = {**observation_frame, **action_frame, "task": single_task}
            normal_dataset.add_frame(normal_frame)
            
            # Pressure dataset frame (add pressure to observation.state)
            pressure_observation_frame = dict(observation_frame)
            
            if OBS_STATE in pressure_observation_frame:
                original_state = np.array(pressure_observation_frame[OBS_STATE], dtype=np.float32)
                extended_state = np.append(original_state, np.float32(pressure))
                pressure_observation_frame[OBS_STATE] = extended_state
            
            pressure_frame = {**pressure_observation_frame, **action_frame, "task": single_task}
            pressure_dataset.add_frame(pressure_frame)

        # Log to visualization
        if display_data:
            try:
                log_pressure_data(obs_processed, act_processed, pressure, frame_count)
            except Exception as e:
                logging.debug(f"Visualization error: {e}")

        frame_count += 1

        # Progress display
        if frame_count % 30 == 0:
            elapsed = time.perf_counter() - start_episode_t
            print(f"  Frames: {frame_count}, Time: {elapsed:.1f}s, Pressure: {pressure:.3f}N", end="\r")

        timestamp = time.perf_counter() - start_episode_t
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)
    
    print()  # New line after progress


@parser.wrap()
def record(cfg: DualRecordConfig) -> tuple[LeRobotDataset, LeRobotDataset]:
    """Record two datasets: normal and with pressure.
    
    This function follows the same structure as lerobot_record.py's record():
    1. Initialize logging and visualization
    2. Create robot and teleoperator
    3. Create processors
    4. Create/load datasets with proper features
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

    # Create normal dataset features (same as lerobot_record.py)
    normal_features = combine_feature_dicts(
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

    # Create pressure-extended features (only difference from normal dataset)
    pressure_features = create_pressure_extended_features(normal_features.copy(), pressure_dim=1)

    # Create or load datasets (following lerobot_record.py pattern)
    num_cameras = len(robot.cameras) if hasattr(robot, 'cameras') and robot.cameras else 0
    
    if cfg.resume:
        # Resume existing datasets
        logging.info("Resuming existing datasets...")
        normal_dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        
        pressure_dataset = LeRobotDataset(
            cfg.pressure_dataset.repo_id,
            root=cfg.pressure_dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        
        # Start image writers if using cameras
        if num_cameras > 0:
            normal_dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            )
            pressure_dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            )
        
        # Sanity check
        sanity_check_dataset_robot_compatibility(normal_dataset, robot, cfg.dataset.fps, normal_features)
        sanity_check_dataset_robot_compatibility(pressure_dataset, robot, cfg.dataset.fps, pressure_features)
        
        logging.info(f"Resumed normal dataset with {normal_dataset.meta.total_episodes} episodes")
        logging.info(f"Resumed pressure dataset with {pressure_dataset.meta.total_episodes} episodes")
    else:
        # Create new datasets (same structure for both)
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        sanity_check_dataset_name(cfg.pressure_dataset.repo_id, cfg.policy)
        
        normal_dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=normal_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        pressure_dataset = LeRobotDataset.create(
            cfg.pressure_dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.pressure_dataset.root,
            robot_type=robot.name,
            features=pressure_features,
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

    # Calculate remaining episodes to record
    if cfg.resume:
        already_recorded = min(normal_dataset.meta.total_episodes, pressure_dataset.meta.total_episodes)
        episodes_to_record = cfg.dataset.num_episodes - already_recorded
        if episodes_to_record <= 0:
            logging.info(f"Already recorded {already_recorded} episodes. Nothing to do.")
            robot.disconnect()
            if teleop is not None:
                teleop.disconnect()
            return normal_dataset, pressure_dataset
        logging.info(f"Already recorded {already_recorded} episodes. Recording {episodes_to_record} more.")
    else:
        already_recorded = 0

    # Recording loop with VideoEncodingManager (same as lerobot_record.py)
    with VideoEncodingManager(normal_dataset), VideoEncodingManager(pressure_dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {normal_dataset.num_episodes}", cfg.play_sounds)
            print(f"\n{'='*50}")
            print(f"--- Episode {normal_dataset.num_episodes} ---")

            # Set episode timeline for Rerun
            if cfg.display_data:
                rr.set_time_sequence("episode", normal_dataset.num_episodes)

            # Main recording loop
            record_loop_with_pressure(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                normal_dataset=normal_dataset,
                pressure_dataset=pressure_dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                has_pressure_sensor=has_pressure_sensor,
            )

            # Reset environment (same logic as lerobot_record.py)
            should_reset = not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            )
            
            if should_reset:
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
                normal_dataset.clear_episode_buffer()
                pressure_dataset.clear_episode_buffer()
                continue

            # Save episodes (same as lerobot_record.py)
            normal_dataset.save_episode()
            pressure_dataset.save_episode()
            recorded_episodes += 1
            
            log_say(f"Episode {recorded_episodes} saved", cfg.play_sounds)

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    # Push to hub (same as lerobot_record.py)
    if cfg.dataset.push_to_hub and recorded_episodes > already_recorded:
        print("Pushing datasets to hub...")
        normal_dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        pressure_dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Recording finished", cfg.play_sounds)
    print(f"\nRecorded {recorded_episodes - already_recorded} new episodes. Total: {recorded_episodes} episodes.")
    print("="*50)
    
    return normal_dataset, pressure_dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()