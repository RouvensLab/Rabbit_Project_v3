from manual_controled_robot import ManualExpert
from envs.force_imitation import CustomEnv
import numpy as np
import time
import json
import os

env_param_kwargs = {
    "ModelType": "SAC",
    "rewards_type": ["stability"],
    "observation_type": ["joint_forces", "joint_angles", "rhythm"],
    "simulation_stepSize": 5,
    "maxSteps": 360*4,
    "restriction_2D": False,
    "terrain_type": "flat",
    #"recorded_movement_file_path_list": r"expert_trajectories\recorded_data_sitting_v1"
}

env = CustomEnv(render_mode="fast", gui=True, **env_param_kwargs)

expert = ManualExpert(sim_freq=5)

seed = None

def convert_data_to_json(data, action, observation, reward):
    pos_array, euler_array, vel_array, angvel_array, joint_angles, joint_torques, joint_velocitys, component_coordinates_world, component_coordinates_local = data
    
    #print(joint_coordinates)
    return [list(action), list(observation), reward, list(pos_array), list(euler_array), list(vel_array), list(angvel_array), list(joint_angles), list(joint_torques), list(joint_velocitys), list(component_coordinates_world), list(component_coordinates_local)]

log_data = {}

#run and record for one episode
for episode in range(1):
    obs, info = env.reset(seed)
    data = env.ROS_Env.get_informations()
    simulation_time = env.ROS_Env.simulation_steps#1 step is 0.01 sec


    done = False
    while not done:
        simulation_time = env.ROS_Env.simulation_steps#1 step is 0.01 sec
        action, state = expert.think_and_respond(obs, None, done)
        obs, reward, terminated, truncated, info = env.step(action)

        data = env.ROS_Env.get_informations()
        log_data[simulation_time] = convert_data_to_json(data, action, obs, reward)

        done = terminated or truncated
env.close()

#save the log data
main_dir = r"expert_trajectories"
name = "realy_fast_springing_v1.json"
log_dir = os.path.join(main_dir, name)


# Assuming log_data is your data containing ndarray objects
#print(log_data)
# Now, you can safely dump converted_log_data to a file
with open(log_dir, 'w') as f:
    json.dump(log_data, f)
