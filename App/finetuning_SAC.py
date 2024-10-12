
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from envs.force_imitation import CustomEnv
from Callbacks import OptimizerCallback

#from envs.global_direct_env_v6 import CustomEnv
# Normalize observations and rewards
def make_env(rank, env_param_kwargs):
    def _init():
        env = CustomEnv(render_mode="fast" if rank==0 else "fast", gui=rank==0, **env_param_kwargs)
        env = Monitor(env)
        return env
    return _init

# def linear_schedule(initial_value):
#     def func(progress_remaining):
#         # Linear schedule: learning rate decreases as progress_remaining decreases
#         return progress_remaining * initial_value
#     return func
    



if __name__ == "__main__":

    trained_model_dir = r"Models\SACReinforceImitation_V2_v7\models\best_model.zip"
    #extract the trained models information
    model_steps = trained_model_dir.split("/")[-1].split("_")[-2]
    trained_model_dir_folder = "\\".join(trained_model_dir.split("/")[:-1])
    print(model_steps)
    print(trained_model_dir_folder)
    tr_model_replay_buffer_dir = r""
    #rollout_data_dir = r"expert_trajectories\recorded_data_sprinting_v1"


    show_last_results = False

    alg_name = "SAC"
    ModellName = "Fine_tuning_SACReinforceImitation_V2_v11"  
    main_dir = r"Models\\" + alg_name + ModellName
    models_dir = os.path.join(main_dir, "models")
    logdir = models_dir+"\\logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    total_timesteps = 40_000_000  # Increase total training steps
    
    MAXFORCE = 2.2
    MAXVELOCITY = 3.8

    env_param_kwargs = {
        "ModelType": "SAC",
        "rewards_type": ["Modified_Peng_et_al_reward"],
        "observation_type": ["joint_forces", "joint_angles", "euler_array", "goal_orientation"],
        "simulation_stepSize": 5,
        "obs_time_space": 2,
        "maxSteps": 360*5,
        "restriction_2D": False,
        "terrain_type": "random_terrain",
        "recorded_movement_file_path_list": [r"expert_trajectories\fast_linear_springing_v1.json", r"expert_trajectories\realy_fast_springing_v1.json"],
        "change_body_params": {"Motors_strength": [MAXFORCE*4, MAXFORCE*4, MAXFORCE, MAXFORCE, MAXFORCE, MAXFORCE,MAXFORCE,MAXFORCE,MAXFORCE*4 ,MAXFORCE*4, MAXFORCE, MAXFORCE], "Motors_velocity": [MAXVELOCITY/4, MAXVELOCITY/4, MAXVELOCITY, MAXVELOCITY, MAXVELOCITY, MAXVELOCITY,MAXVELOCITY,MAXVELOCITY,MAXVELOCITY/4,MAXVELOCITY/4, MAXVELOCITY, MAXVELOCITY]}
    }



    if not show_last_results:

        # Make multiprocess env
        num_cpu = 20  # Adjust the number of CPUs based on your machine
        envs = SubprocVecEnv([make_env(i, env_param_kwargs=env_param_kwargs) for i in range(num_cpu)])
        SEED = None
        load_replay_buffer = False

        

        # Define callbacks
        # eval_envs = SubprocVecEnv([make_env(i+num_cpu, env_param_kwargs=env_param_kwargs) for i in range(1)])
        # eval_callback = EvalCallback(eval_envs, best_model_save_path=models_dir,
        #                             log_path=logdir, eval_freq=10000,
        #                             deterministic=False, render=False,
        #                             n_eval_episodes=5
        #                             )

        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=models_dir,
                                                name_prefix='rl_model', save_replay_buffer=False)
        
        # Create the custom callback
        #optimizer_call = OptimizerCallback(log_dir=logdir, eval_freq=5000, best_model_save_path=models_dir, save_replay_buffer=True, name_prefix="best_rl_model", learningrate_adjustment=False)
        # # check if there are some replay_buffer data to the trained model
        # if rollout_data_dir != "":
        #     #load the rollouts into the model
        #     #save the expert trajectories
        #     exp_rollouts = serialize.load(rollout_data_dir)
        #     print("Rollouts have been added to the replay buffer")


        # Load the trained agent
        model = SAC.load(trained_model_dir, env=envs, tensorboard_log=logdir, verbose=1, device="cuda", print_system_info=True)#, custom_objects={"learning_rate": 1*1e-5})
        #model.set_parameters(load_path_or_dict=trained_model_dir, exact_match=False)
        # model.batch_size = 3000
        # model.learning_rate = 1*1e-5
        #get hyperparameters for the model
        #hyper_params = model.get_parameters()

        # check if there are some replay_buffer data from the trained model
        if tr_model_replay_buffer_dir != r"":
            tr_model_replay_buffer_dir = trained_model_dir_folder+f"\\rl_model_replay_buffer_{model_steps}_steps.pkl"
        if os.path.exists(tr_model_replay_buffer_dir) and load_replay_buffer:
            #load the replay buffer into the model
            model.load_replay_buffer(tr_model_replay_buffer_dir)
            print("Replay buffer loaded")
        else:
            print("No replay buffer loaded")
            load_replay_buffer = False


        #creates a documentation of the model hyperparameter, the environements parameter and other information concearned to the training, in the Model directory.
        with open(main_dir + "/info.txt", "w") as f:
            f.write(f"Model: {alg_name}\n")
            f.write(f"Model Name: {ModellName}\n")
            f.write(f"Number of CPUs: {num_cpu}\n")
            f.write(f"Total Timesteps: {total_timesteps}\n")
            f.write(f"Env Parameters: {env_param_kwargs}\n")
            #f.write(f"Added rollouts: {rollout_data_dir}\n")
            f.write(f"Seed: {SEED}\n\n")

            f.write(f"Trained Model: {trained_model_dir}\n")
            f.write(f"Load Replay Buffer: {load_replay_buffer}\n")
            f.write(f"Replay Buffer Directory: {tr_model_replay_buffer_dir}\n")
            #f.write(f"Hyperparameters: {hyper_params}\n")

            f.write(f"Used_env_class_filename: {"force_imitation_v2"}\n")


        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])#, optimizer_call])

        model.save(f"{models_dir}/final_model")
        model.save_replay_buffer(os.path.join(models_dir, f"rl_model_replay_buffer_{total_timesteps}_steps.pkl"))
        envs.close()
        
    
    else:
        # Load the trained agent
        model = SAC.load(models_dir + "/best_model.zip")

        # Evaluate the agent
        env = CustomEnv(render_mode="fast", gui=True, **env_param_kwargs)
        obs, _ = env.reset()
        for i in range(1000):
            action, info = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
