"""This code is used to evaluate the agents, when they are not learning. So they give theier bigges potential.

you can simpy just evaluate 

"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Callable, List
import json
import numpy as np
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)
from envs.force_imitation import CustomEnv
#from envs.global_direct_env_v6 import CustomEnv
# Normalize observations and rewards
def make_env(rank, env_param_kwargs):
    def _init():
        env = CustomEnv(render_mode="fast" if rank==0 else "fast", gui=rank==0, **env_param_kwargs)
        env = Monitor(env)
        return env
    return _init

# Function to format x-axis in Millions
def episodes(x, pos):
    return f'{int(x)}Ep'

def plot_compare_logdirs(logdirs:List[str], name_list:List[str], colors=List[str], transforming_func:Callable=None, output_dir=None, show=False, Names_in_title=False, graph_type="line"):
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    all_ea_logs = []
    same_metrics = []
    for logdir in logdirs:
        #open the csv file
        ea = pd.read_csv(logdir)

        all_ea_logs.append(ea)
        available_metrics = list(ea.columns)
        if not same_metrics:
            same_metrics = available_metrics
        else:
            same_metrics = list(set(same_metrics).intersection(available_metrics))
        

    for metric in same_metrics:

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if Names_in_title:
            title = metric.replace("/", "_")+"_"+"_".join(name_list)
        else:
            title = metric.replace("/", "_")


        

        for id, ea in enumerate(all_ea_logs):
            if transforming_func:
                steps, values, title = transforming_func(ea)
                ax.plot(steps, values, label=name_list[id], linewidth=2)
            else:
                events = ea[metric]
                steps = list(range(len(events)))
                values = [event for event in events]
                if graph_type == "line":
                    ax.plot(steps, values, label=name_list[id], linewidth=2, color=colors[id])
                    #show the mean value line with the same color as the line
                    ax.axhline(np.mean(values), color=colors[id], linestyle='--', linewidth=1)
                elif graph_type == "bar":
                    ax.bar(name_list[id], np.mean(values), label=name_list[id], color=colors[id])

        # Format x-axis to show steps in millions
        if graph_type == "line":
            # Grid and ticks
            ax.grid(True, which='both', linestyle='--', linewidth=0.7)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in')
            # Set x-axis formatter
            ax.xaxis.set_major_formatter(FuncFormatter(episodes))
        else:
            # change figure size for bar plot to narrow
            fig.set_size_inches(5, 5)
        ax.set_xlabel("Episodes")
        ax.set_ylabel(metric)
        ax.set_title(f"Evaluation: {title}", pad=15)
        # Adjust legend outside the plot
        ax.legend()
        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save the plot or show it
        if output_dir:
            output_path = os.path.join(output_dir, f"{title}_{graph_type}.png")
            fig.savefig(output_path)

        if show:
            plt.show()

        if transforming_func:
            break

def plot_multiBox_compare(list_dictionary, data_types:List[str], env_types:List[str], name_colors=List[str], output_dir=None, show=False, Names_in_title=False):
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    for i, data_dict in enumerate(list_dictionary):
        fig, ax = plt.subplots()

        x = np.arange(len(env_types))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        for attribute, measurement in data_dict.items():
            offset = width * multiplier
            # round the mesurement to 1 decimal
            measurement = [round(val, 0) for val in measurement]
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        
        
        ax.set_xticks(x+width, env_types)
        ax.set_ylabel(data_types[i])
        title = f"Evaluation: {data_types[i]}"
        ax.set_title(title, pad=15)
        # Adjust legend outside the plot
        ax.legend()
        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save the plot or show it
        if output_dir:
            output_path = os.path.join(output_dir, f"{title}.png")
            fig.savefig(output_path)
        if show:
            plt.show()

def evaluate_agents(models_dir_list, env_param_kwargs):
    """Evaluates all the agents in the models_dir_list. They have to be SAC models.
    Args:
        models_dir_list (list): List of directories where the models are saved.
        env_param_kwargs (dict): Dictionary with the environment parameters.

    Returns:
        list: List of the times the agent was terminated or truncated.
        list: List of the rewards the agent got in each episode.
        list: List of the lengths of the episodes.
    """
    # Evaluate the agent. This gives insight into the agent's performance, like how good is the returned reward over the episodes. How often had the agent died, and how often had the agent reached the goal/ truncated.
    eval_episode_num = 50
    env = CustomEnv(render_mode="fast", gui=True, **env_param_kwargs)
    
    Agent_times_terminated_tuncated_list = []
    Agent_ep_rew_list = []
    Agent_ep_len_list = []

    for models_dir, logdir in zip(models_dir_list, models_dir_logs_list):
        model = SAC.load(models_dir + "/best_model.zip")
        times_terminated_tuncated = [0, 0]
        ep_col_reward_list = []
        ep_len_list = []
        for i in range(eval_episode_num):
            collected_reward = 0
            ep_length = 0

            done = False
            obs, _ = env.reset()
            while done == False:
                action, info = model.predict(obs)
                obs, rewards, terminated, truncated, info = env.step(action)
                collected_reward += rewards
                ep_length += 1              
                
                if truncated or terminated:
                    done = True
                    if terminated:
                        times_terminated_tuncated[0] += 1
                    else:
                        times_terminated_tuncated[1] += 1
            
            ep_col_reward_list.append(collected_reward)
            ep_len_list.append(ep_length)


        Agent_times_terminated_tuncated_list.append(times_terminated_tuncated)
        Agent_ep_rew_list.append(ep_col_reward_list)
        Agent_ep_len_list.append(ep_len_list)

    env.close()

    return Agent_times_terminated_tuncated_list, Agent_ep_rew_list, Agent_ep_len_list

def evaluate_agents_in_all_terrains(terrain_type_list, models_dir_list, Models_titles):
    # Evaluate the agents with all the terrains types

    mean_terrain_values_dict_list = []
    for i in range(2):
        mean_value_terrain_dict = {}
        for agent in Models_titles:
            mean_value_terrain_dict[agent] = []
        mean_terrain_values_dict_list.append(mean_value_terrain_dict)
    for terrain_type in terrain_type_list:
        env_param_kwargs = {
            "ModelType": "SAC",
            "rewards_type": ["Modified_Peng_et_al_reward"],
            "observation_type": ["joint_forces", "joint_angles", "euler_array", "goal_orientation"],
            "simulation_stepSize": 5,
            "obs_time_space": 2,
            "maxSteps": 360*5,
            "restriction_2D": False,
            "terrain_type": terrain_type,
            "recorded_movement_file_path_list": [r"expert_trajectories\fast_linear_springing_v1.json", 
                                                r"expert_trajectories\slow_springing_v2.json", 
                                                r"expert_trajectories\fast_right_springing_v2.json",
                                                r"expert_trajectories\fast_curve_springing_v2.json",
                                                r"expert_trajectories\slow_left_springing_v2.json",
                                                r"expert_trajectories\fast_left_springing_v2.json"]
        }


        # Evaluate the agents
        Agent_times_terminated_tuncated_list, Agent_ep_rew_list, Agent_ep_len_list = evaluate_agents(models_dir_list, env_param_kwargs)
        data_list = [Agent_ep_rew_list, Agent_ep_len_list]
        for i, data_type in enumerate(data_list):

            for id, agent in enumerate(Models_titles):
                mean_values_terrain = mean_terrain_values_dict_list[i][agent]
                mean_values_terrain.append(np.mean(data_type[id]))
                mean_terrain_values_dict_list[i][agent] = mean_values_terrain
    
    #save the mean_terrain_values_dict_list as a json file
    with open(os.path.join(data_folder, "mean_terrain_values_dict_list.json"), "w") as f:
        json.dump(mean_terrain_values_dict_list, f)

    #and make a plot
    plot_multiBox_compare(mean_terrain_values_dict_list, ["Reward", "Episode Length"], terrain_type_list, ["blue", "orange", "green"], output_dir=data_folder, show=True)

    return mean_terrain_values_dict_list

        


    




    
def evaluate_agents_in_one_terrain(terrain_type, models_dir_list, ModellNames):
    env_param_kwargs = {
        "ModelType": "SAC",
        "rewards_type": ["Modified_Peng_et_al_reward"],
        "observation_type": ["joint_forces", "joint_angles", "euler_array", "goal_orientation"],
        "simulation_stepSize": 5,
        "obs_time_space": 2,
        "maxSteps": 360*5,
        "restriction_2D": False,
        "terrain_type": terrain_type,
        "recorded_movement_file_path_list": [r"expert_trajectories\fast_linear_springing_v1.json", 
                                            r"expert_trajectories\slow_springing_v2.json", 
                                            r"expert_trajectories\fast_right_springing_v2.json",
                                            r"expert_trajectories\fast_curve_springing_v2.json",
                                            r"expert_trajectories\slow_left_springing_v2.json",
                                            r"expert_trajectories\fast_left_springing_v2.json"]
    }

    # Evaluate the agents
    Agent_times_terminated_tuncated_list, Agent_ep_rew_list, Agent_ep_len_list = evaluate_agents(models_dir_list, env_param_kwargs)

    for i, ModellName in enumerate(ModellNames):
        data = {"times_terminated": Agent_times_terminated_tuncated_list[i][0], "times_truncated": Agent_times_terminated_tuncated_list[i][1], "ep_rewards": Agent_ep_rew_list[i], "ep_lengths": Agent_ep_len_list[i]}
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_folder, f"{ModellName}_eval_results.csv"), index=False)

    # Plot the results
    result_save_dirs = []
    for i, ModellName in enumerate(ModellNames):
        result_save_dirs.append(os.path.join(data_folder, f"{ModellName}_{"uneven_terrain"}_eval_results.csv"))
    plot_compare_logdirs(result_save_dirs, ["Umgebung 1", "Umgebung 2", "Umgebung 3"], colors=["blue", "orange", "green"], show=True, output_dir=data_folder, Names_in_title=False, graph_type="line")


if __name__ == "__main__":

    #tr_model_replay_buffer_dir = r"expert_trajectories\recorded_data_sprinting_v1"


    ModellNames = ["SACReinforceImitation_V2_v8", "SACReinforceImitation_V2_v6", "SACReinforceImitation_V2_v7"]
    Models_titles = ["Umgebung 1", "Umgebung 2", "Umgebung 3"]
    models_dir_list = []
    models_dir_logs_list = []
    for ModellName in ModellNames:
        main_dir = r"Models\\" + ModellName
        models_dir_list.append(os.path.join(main_dir, "models"))
        #models_dir_logs_list.append(models_dir_list[-1]+"\\logs")

    #Save the results to the data folder
    data_folder = r"App\graphs\data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)


    # #to just get graphs of one terrain where all the agents in the list are evaluated
    # evaluate_agents_in_one_terrain("uneven_terrain", models_dir_list, ModellNames)

    
    # #get a boxplot of all the terrains
    # #evaluate_agents_in_all_terrains(terrain_type_list)
    terrain_type_list = ["flat", "uneven_terrain", "random_terrain"]
    # evaluate_agents_in_all_terrains(terrain_type_list, models_dir_list, Models_titles)
    
    #open the json file
    with open(os.path.join(data_folder, "mean_terrain_values_dict_list.json"), "r") as f:
        mean_terrain_values_dict_list = json.load(f)
    #and make a plot
    plot_multiBox_compare(mean_terrain_values_dict_list, ["Reward", "Episode Length"], terrain_type_list, ["blue", "orange", "green"], output_dir=data_folder, show=True)


    
