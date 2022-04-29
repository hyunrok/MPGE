from gym.envs.registration import register
from grounding_codes import *
from gym.envs.mujoco import half_cheetah
from gym.envs.mujoco import hopper

register(
    id='HopperBroken-v2',
    entry_point='grounding_codes.envs.mujoco:BrokenJoint',
    max_episode_steps=1000,
    reward_threshold=3800.0,
    kwargs={
        'env': hopper.HopperEnv(),
        'broken_joint': 0,
    }
)

register(
    id='InvertedPendulumPositiveSkew-v2',
    entry_point='grounding_codes.envs.mujoco:InvertedPendulumPositiveSkewEnv',
    max_episode_steps=1000,
    reward_threshold=1000.0,
    )

register(
    id='HalfCheetahModified-v2',
    entry_point='grounding_codes.envs.mujoco:HalfCheetahModifiedEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetahBroken-v2',
    entry_point='grounding_codes.envs.mujoco:BrokenJoint',
    max_episode_steps=1000,
    reward_threshold=4800.0,
    kwargs={
        'env': half_cheetah.HalfCheetahEnv(),
        'broken_joint': 0,
    }
)