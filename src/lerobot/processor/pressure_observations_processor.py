# filepath: /Users/hiroaki.ishikawa/Documents/src/so101/lerobot/src/lerobot/processor/pressure_observations_processor.py
#!/usr/bin/env python

"""
圧力センサープロセッサ: observation.stateにグリッパー圧力値を追加

出力:
- observation.state: 元の状態（既存スクリプト用）
- observation.state_with_pressure: 圧力値を追加した状態
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.pipeline import (
    ObservationProcessorStep,
    PipelineFeatureType,
    ProcessorStepRegistry,
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.constants import OBS_STATE

# 新しい観測キー
OBS_STATE_WITH_PRESSURE = "observation.state_with_pressure"


@dataclass
@ProcessorStepRegistry.register(name="pressure_sensor_processor")
class PressureSensorProcessorStep(ObservationProcessorStep):
    """
    グリッパーの圧力センサー値をobservationに追加するプロセッサ。
    
    SO101Leaderのread_pressure_sensor()メソッドから圧力値を取得し、
    元のobservation.stateを保持しつつ、
    observation.state_with_pressureに圧力値を追加した状態を作成。
    
    Args:
        teleop: 圧力センサーを持つテレオペレーターインスタンス (SO101Leader)
    """
    
    teleop: Teleoperator | None = None
    _pressure_dim: int = field(default=1, init=False, repr=False)  # グリッパー圧力は1次元
    
    def _read_pressure(self) -> torch.Tensor:
        """テレオペレーターから圧力値を読み取る"""
        if self.teleop is None:
            return torch.zeros(self._pressure_dim, dtype=torch.float32)
        
        try:
            # SO101Leaderのread_pressure_sensor()を使用
            if hasattr(self.teleop, 'read_pressure_sensor'):
                pressure_value = self.teleop.read_pressure_sensor()
                return torch.tensor([pressure_value], dtype=torch.float32)
            else:
                return torch.zeros(self._pressure_dim, dtype=torch.float32)
        except Exception as e:
            print(f"圧力読み取りエラー: {e}")
            return torch.zeros(self._pressure_dim, dtype=torch.float32)
    
    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        観測データにグリッパー圧力情報を追加。
        
        元のobservation.stateは保持し、
        新たにobservation.state_with_pressureを追加。
        """
        current_state = obs.get(OBS_STATE)
        if current_state is None:
            return obs
        
        # 圧力値を取得
        pressure_tensor = self._read_pressure()
        
        # 次元を合わせる
        if current_state.dim() == 2:
            # (batch, state_dim) の場合
            pressure_tensor = pressure_tensor.unsqueeze(0)
        
        # 圧力を追加した拡張状態を作成
        extended_state = torch.cat([current_state, pressure_tensor], dim=-1)
        
        # 新しいobsを作成（元のstateは保持）
        new_obs = dict(obs)
        new_obs[OBS_STATE] = current_state  # 元のまま
        new_obs[OBS_STATE_WITH_PRESSURE] = extended_state  # 圧力追加版
        
        return new_obs
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        feature定義を更新して、圧力付き状態を追加。
        """
        obs_features = features.get(PipelineFeatureType.OBSERVATION, {})
        
        if OBS_STATE in obs_features:
            original_feature = obs_features[OBS_STATE]
            
            # 元の状態の次元数を取得
            if original_feature.shape:
                original_dim = original_feature.shape[0]
            else:
                original_dim = 0
            
            # 拡張した次元数 (元の次元 + グリッパー圧力1次元)
            extended_dim = original_dim + self._pressure_dim
            
            # 新しいfeatureを追加
            obs_features[OBS_STATE_WITH_PRESSURE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(extended_dim,),
            )
            
            features[PipelineFeatureType.OBSERVATION] = obs_features
        
        return features