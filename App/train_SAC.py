from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from envs.force_imitation import CustomEnv
# Normalize observations and rewards
def make_env(rank, env_param_kwargs):
    def _init():
        env = CustomEnv(render_mode="fast" if rank==0 else "fast", gui=rank==0, **env_param_kwargs)
        env = Monitor(env)
        return env
    return _init

class BabyModeFreedomCallback(BaseCallback):
    """
    A custom callback that logs the 'baby_mode_freedom' variable from the environment.
    """

    def __init__(self, verbose=0):
        super(BabyModeFreedomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the 'baby_mode_freedom' variable using the 'get_attr' method
        # baby_mode_freedom_values = self.training_env.get_attr('baby_mode_freedom', indices=None)

        # # Log the 'baby_mode_freedom' value
        # self.logger.record('env/baby_mode_freedom_mean', np.mean(baby_mode_freedom_values))
        # self.logger.record('env/baby_mode_freedom_max', np.max(baby_mode_freedom_values))
        # self.logger.record('env/baby_mode_freedom_min', np.min(baby_mode_freedom_values))

        return True

if __name__ == "__main__":

    show_last_results = False

    #tr_model_replay_buffer_dir = r"expert_trajectories\recorded_data_sprinting_v1"


    alg_name = "SAC"
    ModellName = "ReinforceImitation_V2_v9"
    main_dir = r"Models\\" + alg_name + ModellName
    models_dir = os.path.join(main_dir, "models")
    logdir = models_dir+"\\logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    total_timesteps = 50_000_000  # Increase total training steps

    env_param_kwargs = {
        "ModelType": "SAC",
        "rewards_type": ["Modified_Peng_et_al_reward"],
        "observation_type": ["joint_forces", "joint_angles", "euler_array", "goal_orientation"],
        "simulation_stepSize": 5,
        "obs_time_space": 2,
        "maxSteps": 360*5,
        "restriction_2D": False,
        "terrain_type": "random_terrain",
        "recorded_movement_file_path_list": [r"expert_trajectories\fast_linear_springing_v1.json", 
                                             r"expert_trajectories\slow_springing_v2.json", 
                                             r"expert_trajectories\fast_right_springing_v2.json",
                                             r"expert_trajectories\fast_curve_springing_v2.json",
                                             r"expert_trajectories\slow_left_springing_v2.json",
                                             r"expert_trajectories\fast_left_springing_v2.json"]
    }

    hyper_params = {
        "learning_rate": 5*1e-5,#2*1e-5=0.00002
        "batch_size": 3000,
        "buffer_size": 500_000,
        "policy_kwargs": dict(net_arch=dict(pi=[512, 256], qf=[512, 256])),
        "gamma" : 0.95
        #"action_noise": 0.01,

    }


    if not show_last_results:

        # Make multiprocess env
        num_cpu = 3 # Adjust the number of CPUs based on your machine
        envs = SubprocVecEnv([make_env(i, env_param_kwargs=env_param_kwargs) for i in range(num_cpu)])

        #creates a documentation of the model hyperparameter, the environements parameter and other information concearned to the training, in the Model directory.
        with open(main_dir + "/info.txt", "w") as f:
            f.write(f"Model: {alg_name}\n")
            f.write(f"Model Name: {ModellName}\n")
            f.write(f"Number of CPUs: {num_cpu}\n")
            f.write(f"Total Timesteps: {total_timesteps}\n")
            f.write(f"Env Parameters: {env_param_kwargs}\n")
            f.write(f"Hyperparameters: {hyper_params}\n")

            f.write(f"Used_env_class_filename: {"force_imitation_v2"}\n")
            #f.write(f"Added rollouts: {tr_model_replay_buffer_dir}\n")
        

        # Define callbacks
        eval_env = CustomEnv(render_mode="fast", gui=False, **env_param_kwargs)
        eval_env = Monitor(eval_env)  # Wrap evaluation environment with Monitor

        eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                        log_path=logdir, eval_freq=10000,
                        deterministic=False, render=False,
                        n_eval_episodes=5
                        )

        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=models_dir,
                            name_prefix='rl_model')
        # Create the custom callback
        baby_mode_freedom_callback = BabyModeFreedomCallback()

        # # check if there are some replay_buffer data to the trained model
        # if tr_model_replay_buffer_dir != "":
        #     #load the rollouts into the model
        #     #save the expert trajectories
        #     exp_rollouts = serialize.load(tr_model_replay_buffer_dir)
        #     print("Rollouts are added to the replay buffer")

        # #load pre-trained PPO policy
        model = SAC("MlpPolicy", envs,
                    tensorboard_log=logdir,
                    device='cuda',
                    verbose=1,
                    # rollout_buffer_class=startReplayBuffer,
                    # rollout_buffer_kwargs=dict(demonstrations=exp_rollouts, demonstrations_wight=13),
                    **hyper_params
                    )
        
        #load trained SAC policy
        #model = SAC.load(r"Models\SACReinforceImitation_v10\models\best_model.zip", env=envs, tensorboard_log=logdir)

        

        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False,
                    tb_log_name=alg_name, callback=[eval_callback, checkpoint_callback, baby_mode_freedom_callback])

        model.save(f"{models_dir}/final_model")
        model.save_replay_buffer(os.path.join(models_dir, f"rl_model_replay_buffer_{total_timesteps}_steps.pkl"))

        envs.close()
    
    else:
        # Load the trained agent
        model = SAC.load(models_dir + "/best_model.zip")

        # Evaluate the agent
        env = CustomEnv(render_mode="human", gui=True, **env_param_kwargs)
        obs, _ = env.reset()
        for i in range(1000):
            action, info = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            if truncated:
                obs, _ = env.reset()

        env.close()
