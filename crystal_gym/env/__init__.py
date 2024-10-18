import gymnasium
from crystal_gym.env.crystal_env import CrystalGymEnv

gymnasium.register(
        id='CrystalGymEnv-v0',
        entry_point='crystal_gym.env.crystal_env:CrystalGymEnv',
    )