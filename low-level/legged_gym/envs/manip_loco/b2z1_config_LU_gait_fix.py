"""LU configuration with corrected gait reward signs for a new training run.

The contact reward functions already return nonpositive penalties. Positive
scales penalize swing-foot contact and stance-foot sliding. Keep their original
magnitude so this experiment isolates the sign correction.

Inherit the updated LU command deadzone (0.15 m/s, 0.35 rad/s). Original LU
checkpoints used 0.2 m/s and 0.5 rad/s. Changing rewards at playback cannot
repair learned weights.
"""

from legged_gym.envs.manip_loco.b2z1_config_LU import (
    B2Z1RoughCfg as _LUEnvCfg,
    B2Z1RoughCfgPPO as _LUTrainCfg,
)


class B2Z1RoughCfg(_LUEnvCfg):
    class rewards(_LUEnvCfg.rewards):
        class scales(_LUEnvCfg.rewards.scales):
            tracking_contacts_shaped_force = 0.2
            tracking_contacts_shaped_vel = 0.2


class B2Z1RoughCfgPPO(_LUTrainCfg):
    pass
