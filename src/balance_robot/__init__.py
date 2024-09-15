from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

register(
    id="Env01-v1",
    entry_point="balance_robot.envs.env01_v1:Env01",
    max_episode_steps=2000,
    reward_threshold=950.0,
)

