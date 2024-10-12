from manual_controled_robot import ManualExpert
from App.envs.old.force_imitation_v1 import CustomEnv
import numpy as np
import time
import json
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import numpy as np
import imitation.data.rollout as rollout
from imitation.data import serialize

# Normalize observations and rewards
def make_env(rank, env_param_kwargs):
    def _init():
        env = CustomEnv(render_mode="fast" if rank==0 else "fast", gui=rank==0, **env_param_kwargs)
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    seed = 42

    env_param_kwargs = {
       "ModelType": "SAC",
        "rewards_type": ["exploration", "center_mass_height", "force_imitation"],
        "observation_type": ["joint_forces", "joint_angles", "goal_orientation", "rhythm"],
        "simulation_stepSize": 10,
        "restriction_2D": True,
        "goal_reward": 13,
        "baby_mode_freedom": 1.0,
        "recorded_movement_file_path": r"expert_trajectories\recorded_data_sprinting_v2.json"
    }

    num_cpu = 5

    expert = ManualExpert(sim_freq=10)

    envs = SubprocVecEnv([make_env(i, env_param_kwargs=env_param_kwargs) for i in range(num_cpu)])
    # Generate the expert trajectories with rollout.rollout(...)
    exp_trajec = rollout.rollout(
        policy=expert.think_response_with_ndarray,  # Ensure this is a callable
        venv=envs,
        sample_until=rollout.make_sample_until(min_episodes=50),  # Correct stopping condition
        rng=np.random.RandomState(seed),
        unwrap=False # Keep the data in the same format as the rollout
    )


    #save the log data

    main_dir = r"expert_trajectories"
    name = "recorded_data_sprinting_v1"
    log_dir = os.path.join(main_dir, name)

    #save the expert trajectories
    serialize.save(log_dir, exp_trajec)




