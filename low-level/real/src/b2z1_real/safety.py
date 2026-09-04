"""Safety checks shared by the ROS node and offline tests."""

from __future__ import annotations

from enum import Enum
from typing import Mapping, Optional, Sequence

import numpy as np


class ControlState(str, Enum):
    DISARMED = "DISARMED"
    ARMING = "ARMING"
    ACTIVE = "ACTIVE"
    FAULT = "FAULT"


class SafetyGate:
    def __init__(self, config: Mapping[str, object]):
        self.state_timeout = float(config.get("state_timeout", 0.10))
        self.max_roll = float(config.get("max_roll", 0.50))
        self.max_pitch = float(config.get("max_pitch", 0.50))
        self.max_leg_temperature = float(config.get("max_leg_temperature", 80.0))
        self.max_arm_temperature = float(config.get("max_arm_temperature", 75.0))
        self.leg_lower = np.asarray(config["leg_lower_limits"], dtype=np.float64)
        self.leg_upper = np.asarray(config["leg_upper_limits"], dtype=np.float64)
        self.arm_lower = np.asarray(config["arm_lower_limits"], dtype=np.float64)
        self.arm_upper = np.asarray(config["arm_upper_limits"], dtype=np.float64)
        for name, values, size in (
            ("leg_lower_limits", self.leg_lower, 12),
            ("leg_upper_limits", self.leg_upper, 12),
            ("arm_lower_limits", self.arm_lower, 6),
            ("arm_upper_limits", self.arm_upper, 6),
        ):
            if values.shape != (size,):
                raise ValueError(f"{name} must have {size} entries")

    def check(
        self,
        now: float,
        b2_received_at: Optional[float],
        z1_received_at: Optional[float],
        rpy: Sequence[float],
        leg_position: Sequence[float],
        arm_position: Sequence[float],
        leg_temperature: Sequence[float],
        arm_temperature: Sequence[float],
        b2_valid: bool,
        z1_valid: bool,
    ) -> Optional[str]:
        if b2_received_at is None or now - b2_received_at > self.state_timeout:
            return "B2 state timeout"
        if z1_received_at is None or now - z1_received_at > self.state_timeout:
            return "Z1 state timeout"
        if not b2_valid:
            return "B2 state or CRC invalid"
        if not z1_valid:
            return "Z1 motor state invalid"

        rpy = np.asarray(rpy, dtype=np.float64)
        leg_position = np.asarray(leg_position, dtype=np.float64)
        arm_position = np.asarray(arm_position, dtype=np.float64)
        if not (np.all(np.isfinite(rpy)) and np.all(np.isfinite(leg_position)) and np.all(np.isfinite(arm_position))):
            return "non-finite robot state"
        if abs(rpy[0]) > self.max_roll:
            return f"roll limit exceeded: {rpy[0]:.3f} rad"
        if abs(rpy[1]) > self.max_pitch:
            return f"pitch limit exceeded: {rpy[1]:.3f} rad"
        if np.any(leg_position < self.leg_lower) or np.any(leg_position > self.leg_upper):
            return "B2 joint position outside configured limits"
        if np.any(arm_position < self.arm_lower) or np.any(arm_position > self.arm_upper):
            return "Z1 joint position outside configured limits"
        if len(leg_temperature) and np.max(leg_temperature) >= self.max_leg_temperature:
            return "B2 motor temperature limit exceeded"
        if len(arm_temperature) and np.max(arm_temperature) >= self.max_arm_temperature:
            return "Z1 motor temperature limit exceeded"
        return None
