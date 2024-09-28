from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

register(
    id="Env01-v1",
    entry_point="balance_robot.envs.env01_v1:Env01",
    max_episode_steps=6000,
    reward_threshold=6000,
)

register(
    id="Env01-v2",
    entry_point="balance_robot.envs.env01_v2:Env01_v2",
    max_episode_steps=6000,
    reward_threshold=6000,
)

register(
    id="Env02-v1",
    entry_point="balance_robot.envs.env02_v1:Env02",
    max_episode_steps=6000,
    reward_threshold=6000,
)

register(
    id="Env03-v1",
    entry_point="balance_robot.envs.env03_v1:Env03",
    max_episode_steps=6000,
    reward_threshold=6000,
)

register(
    id="Env03-v2",
    entry_point="balance_robot.envs.env03_v2:Env03_v2",
    max_episode_steps=1200,
    reward_threshold=6000,
)
