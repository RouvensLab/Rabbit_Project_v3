import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

import sys
import os

# Assuming Bunny_Project_v2 is the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

from envs.check_env import ROS_env

#import matplotlib.pyplot as plt

import numpy as np
import time
import math
import json






class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface.
    
    Is optimized to imitate the movement of a expert bunny hopper.
    Is also less data consuming and faster than the CustomEnv_v3.

    Parameters:
    ModelType: str: The type of model that is used to train the robot. Can be "PPO2" or "SAC".
    rewards_type: list: The types of rewards that are used to train the robot. Can be "imitation", "stability", "wrong_movements", "efficiency".
    observation_type: list: The types of observations that are used to train the robot. Can be "euler_array", "vel_array", "joint_angles", "joint_forces", "goal_orientation", "feet_forces", "rhythm".
    render_mode: str: The mode in which the robot is rendered. Can be "human" or "fast".
    gui: bool: If the GUI should be shown or not.
    gym_type: bool: If the environment should have the old gym structure or the new gymnasium structure.
    Horizon_Length: bool: If the environment should have a horizon length or not. Means, when true the robot will stop always after a certain amount of steps and can't die before.
    simulation_stepSize: int: The amount of steps the simulation should take before the next action is taken. default is 5Hz.
    
    """
    
    def __init__(self, ModelType, rewards_type=["stability", "x_vel", "contact_cost", "life_time", "force_imitation", "exploration"], 
                 observation_type=["euler_array", "vel_array", "joint_angles", "joint_forces", "goal_orientation", "feet_forces", "rhythm"], 
                 render_mode="human", 
                 gui=True, 
                 gym_type=False, 
                 Horizon_Length = True,
                 simulation_stepSize = 5,#steps of 0.05s
                 obs_time_space = 2, #in s
                 maxSteps = 366*5,
                 restriction_2D = False,
                 terrain_type = "uneven_terrain",
                 recorded_movement_file_path_list = [r"expert_trajectories\fast_linear_springing_v1.json", r"expert_trajectories\fast_right_springing_v1.json", r"expert_trajectories\fast_left_springing_v1.json"],
                 real_robot = False,
                 change_body_params = None
                 ):
        super().__init__()
        self.ModelType = ModelType
        
        self.RewardsType = rewards_type
        self.ObservationType = observation_type
        self.render_mode = render_mode
        self.gym_type = gym_type
        self.Horizon_Length = Horizon_Length
        self.simulation_stepsize = simulation_stepSize
        self.GUI = gui
        self.record_BodyPos = False
        self._reset_noise_scale = 1e-1
        self.restriction_2D = restriction_2D

        self.maxSteps = maxSteps
        self.real_robot = real_robot
        self.robot_precise_mode = False

        # Frame stacking parameters
        self.n_stack = 10  # Number of frames to stack
        self.stacked_frames = deque(maxlen=self.n_stack)
        self.obs_time_space = obs_time_space #in s


        






        # Initialize the simulation
        self.status_types = ["position", "orientation", "linear_velocity", "angular_velocity", "joint_angles", "joint_torques", "joint_velocitys", "component_coordinates_world", "component_coordinates_local"]
        self.ROS_Env = ROS_env(gui=self.GUI, simulation_speed=self.render_mode,
                               status_types=self.status_types, terrain_type=terrain_type, hung=False, change_body_params=change_body_params)
        #self.ROS_Env.reset()

        #initialize the real robot
        if self.real_robot:
            # Assuming Bunny_Project_v2 is the project root directory
            # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
            # sys.path.append(project_root)
            from envs.robot_env import Robot_env
            self.Robot_Env = Robot_env(gui=self.GUI, simulation_speed=self.render_mode, status_types=self.status_types)


        

        #rhytm values
        self.rhythm = 0


        if "force_imitation" in self.RewardsType or "Peng_et_al_reward" in self.RewardsType or "multi_task_goal" in self.RewardsType or "Modified_Peng_et_al_reward" in self.RewardsType:
            #To show the expert joints coordinates
            self.recorded_motions_list = []
            for recorded_movement_file_path in recorded_movement_file_path_list:
                with open(recorded_movement_file_path, "r") as file:
                    self.recorded_movement = json.load(file)
                #just get the nessesary informations
                self.expert_states_types = {
                    "action": True,
                    "observation": False,
                    "reward": True,
                    "pos_array": True,
                    "euler_array": True,
                    "vel_array": True,
                    "angvel_array": True,
                    "joint_angles": True,
                    "joint_torques": True,
                    "joint_velocitys": True,
                    "component_coordinates_world": True,
                    "component_coordinates_local": False
                }
                self.nessesary_infromation_values = [0, 3, 4, 5, 6, 7,8,9, 10]
                self.recorded_motions_list.append({int(key): [value[i] for i in self.nessesary_infromation_values] for key, value in self.recorded_movement.items()})
                self.recorded_movement.clear()
                

        if "multi_task_goal" in self.RewardsType:
            #task: follow a certain point in the room, for futher integration to use joysticks to control the robot
            optimal_hight = 0.1
            self.goal_time_spaces = 1 #s
            self.goal_min_max_pace = 0.1, 0.3 #m/s
            self.min_max_path_distance = self.goal_time_spaces*self.goal_min_max_pace[0], self.goal_time_spaces*self.goal_min_max_pace[1]
            #generating a random path with multiple vectors
            #vectors have a length of 0.01 to 0.1 and a angle difference from the previous vector of 0 to 45 degrees
            self.direction_vec = np.array([self.goal_min_max_pace[1], 0, 0])
            self.goal_pos_vec = np.array([0, 0, optimal_hight])

        #timeline of joint_velocities
        if "efficiency" in self.RewardsType:
            self.joint_velocities = deque(maxlen=10)
            self.joint_velocities.append([0 for i in range(self.ROS_Env.numMotors)])

        observation_type_sizes = {
            "euler_array": 3,
            "vel_array": 3,
            "joint_angles": self.ROS_Env.numMotors,
            "angvel_array": 3,
            "joint_forces": self.ROS_Env.numMotors,
            "feet_forces": 4,
            "goal_orientation": 2,
            "rhythm": 1
        }
        observation_size = sum([observation_type_sizes[obs] for obs in self.ObservationType])
        self.action_Motor_size = self.ROS_Env.numMotors -2 -2#there are no actionspace for the foots and the symetric spine joints
        self.action_size = self.action_Motor_size
        if "rhythm" in self.ObservationType:
            #observation_size += 1
            self.action_size +=1

        print("observation_size:", observation_size)
        print("action_size:", self.action_size)


        if self.ModelType == "PPO2":
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_size,), dtype=np.float16)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(observation_size * self.n_stack,), dtype=np.float32)

        elif self.ModelType == "SAC":
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_size,), dtype=np.float16)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(observation_size * self.n_stack,), dtype=np.float32)


    def get_new_path(self):
        #check if it is the for a new time space
        if self.ROS_Env.simulation_steps % (self.goal_time_spaces*100) == 0:
            #print(self.ROS_Env.simulation_steps)
            #generating a random path with multiple vectors
            #vectors have a length of 0.01 to 0.1 and a angle difference from the previous vector of 0 to 45 degrees
            start_vec = self.direction_vec
            self.current_pos_vec = self.goal_pos_vec

            start_vec_ang = np.clip(math.atan2(start_vec[1], start_vec[0]), -math.pi, math.pi)
            vec_length = np.random.uniform(self.min_max_path_distance[0], self.min_max_path_distance[1])
            vec_angle = start_vec_ang + np.random.uniform(-math.pi/5, math.pi/5)
            self.direction_vec = np.array([vec_length*math.cos(vec_angle), vec_length*math.sin(vec_angle), 0])
            self.goal_pos_vec = self.current_pos_vec + self.direction_vec
            #self.ROS_Env.show_TextPoint(self.goal_pos_vec, str(start_vec_ang))
            #print("Angles", vec_angle/math.pi*180, start_vec_ang/math.pi*180)

            return self.current_pos_vec, self.goal_pos_vec
        else:
            return self.current_pos_vec, self.goal_pos_vec


    def get_observation(self, information):
        pos_array, euler_array, vel_array, angvel_array, joint_angles, joint_torques, joint_velocitys, component_coordinates_world, component_coordinates_local = information
        observation = []
        
        if "euler_array" in self.ObservationType:
            # Flatten each part of the observation into a 1D array
            euler_array = np.array(euler_array) / (2*math.pi)
            observation = np.concatenate([observation, euler_array])
        
        if "vel_array" in self.ObservationType:
            vel_array = np.clip(vel_array / 100, -1, 1)
            observation = np.concatenate([observation, vel_array])

        if "angvel_array" in self.ObservationType:
            angvel_array = np.clip(angvel_array / (math.pi * 2), -1, 1)
            observation = np.concatenate([observation, angvel_array])

        if "joint_angles" in self.ObservationType:
            joint_angles = [joint_angles[i] / (math.pi * 2) for i in range(self.ROS_Env.numMotors)]
            observation = np.concatenate([observation, joint_angles])
        

        if "joint_forces" in self.ObservationType:
            joint_forces = [np.clip(np.sum(joint_torques[i][:-3]), -1, 1) for i in range(self.ROS_Env.numMotors)]
            observation = np.concatenate([observation, joint_forces])
            #print("joint_forces:", joint_forces)

        # if "feet_forces" in self.ObservationType:
        #     foot_contacts_ = [1 if len(foot_contacts[i]) > 0 else 0 for i in range(4)]
        #     observation = np.concatenate([observation, np.clip(foot_contacts_, 0, 1)])
        #     #print("joint_forces:", foot_contacts_)

        if "goal_orientation" in self.ObservationType:
            if "Peng_et_al_reward" in self.RewardsType or "Modified_Peng_et_al_reward" in self.RewardsType:
                #get the goal orientation from the expert in the future
                exp_coord = self.future_expert_trajectory[1]
                #get the current orientation angle
                orientation_vec = np.array(exp_coord)-np.array(pos_array)
                #calculate the angle between the orientation vector and the x-axis
                goal_orientation = math.atan2(orientation_vec[1], orientation_vec[0])
                goal_agent_angle = goal_orientation/(math.pi) - euler_array[2]
                #normalize the goal_orientation
                goal_orientation_angle = np.clip(goal_agent_angle/10, -1, 1)
                #calculate the lenth of the orientation vector to get a feeling for the distance to the goal
                goal_distance = np.clip(np.linalg.norm(orientation_vec), 0, 1)
                self.calculated_goal_dist = [goal_orientation_angle, goal_distance]
                if self.GUI:
                    self.ROS_Env.show_linked_vectors([pos_array, orientation_vec])
                    #print("goal_orientation_angle:", goal_orientation_angle, "goal_distance:", goal_distance)
                observation = np.concatenate([observation, self.calculated_goal_dist])
            
            elif "multi_task_goal" in self.RewardsType:
                #task: follow a certain point in the room, for futher integration to use joysticks to control the robot
                #get the goal orientation: either the goal_point is radndomly generated or it is given by the joystick
                current_pos, new_pos = self.get_new_path()
                #get the current orientation angle
                orientation_vec = np.array(new_pos)-np.array(pos_array)#This is wront like above in the trained model SAC_V2_v1 (fist big success)
                #calculate the angle between the orientation vector and the x-axis
                goal_orientation = math.atan2(orientation_vec[1], orientation_vec[0])
                goal_agent_angle = goal_orientation/(math.pi) - euler_array[2]
                #normalize the goal_orientation
                goal_orientation_angle = np.clip(goal_agent_angle, -1, 1)
                #calculate the lenth of the orientation vector to get a feeling for the distance to the goal
                goal_distance = np.clip(np.linalg.norm(orientation_vec)/10, 0, 1)
                self.calculated_goal_dist = [goal_orientation_angle, goal_distance]
                observation = np.concatenate([observation, self.calculated_goal_dist])

                if self.GUI:
                    #show the path
                    self.ROS_Env.show_linked_vectors([current_pos, self.direction_vec])
                    #show the vector addet to the robots pos
                    self.ROS_Env.show_linked_vectors([pos_array, orientation_vec])


            else:
                #Error: To use the goal_orientation the force_imitation reward type must be used
                observation = np.concatenate([observation, [0, 0]])
                #raise ValueError("To use the goal_orientation the force_imitation reward type must be used")


        if "rhythm" in self.ObservationType:
            observation = np.concatenate([observation, [self.rhythm]])

        return observation




    def calculate_reward(self, informations, actions, observation, expert_trajectory):
        pos_array, euler_array, vel_array, angvel_array, joint_angles, joint_torques, joint_velocitys, component_coordinates_world, component_coordinates_local = informations
        
        sum_reward = 0
        
        #show which link_index is which link
        #self.show_link_index_pos_in_sim(body_positions, pos_array)

        if "multi_task_goal" in self.RewardsType:
            # Extract expert trajectory data
            exp_actions, exp_pos, exp_euler_array, exp_vel_array, exp_angvel_array, exp_joint_angles, exp_joint_torques, exp_joint_velocitys, exp_component_coordinates_world = expert_trajectory
            
            # Weights from the provided reward function
            wp, wv, wrp = 0.35, 0.05, 0.65

            # Pose Reward (rp_t)
            distance = np.sum([self.euclidean_distance(exp_joint_angles[j], joint_angles[j])**2 for j in range(len(joint_angles))])
            rp_t = np.exp(-5 * distance)
            sum_reward += wp * rp_t

            # Velocity Reward (rv_t)
            distance = np.sum([self.euclidean_distance(exp_joint_velocitys[j], joint_velocitys[j])**2 for j in range(len(joint_velocitys))])
            rv_t = np.exp(-0.1 * distance)
            sum_reward += wv * rv_t

            # Root Pose Reward (rrp_t)
            root_position_distance1 = self.euclidean_distance(self.goal_pos_vec, pos_array)
            root_position_distance2 = self.euclidean_distance(self.current_pos_vec, pos_array)
            root_rotation_distance = self.calculated_goal_dist[0]
            rrp_t = np.exp(-10 * root_position_distance1**2 + -10 * root_position_distance2**2 - 5 * root_rotation_distance**2)
            sum_reward += wrp * rrp_t

        if "Peng_et_al_reward" in self.RewardsType:
            # Extract expert trajectory data
            exp_actions, exp_pos, exp_euler_array, exp_vel_array, exp_angvel_array, exp_joint_angles, exp_joint_torques, exp_joint_velocitys, exp_component_coordinates_world = expert_trajectory
            
            # Weights from the provided reward function
            wp, wv, we, wrp, wrv = 0.5, 0.05, 0.2, 0.15, 0.1

            # Pose Reward (rp_t)
            distance = np.sum([self.euclidean_distance(exp_joint_angles[j], joint_angles[j])**2 for j in range(len(joint_angles))])
            rp_t = np.exp(-5 * distance)
            sum_reward += wp * rp_t

            # Velocity Reward (rv_t)
            distance = np.sum([self.euclidean_distance(exp_joint_velocitys[j], joint_velocitys[j])**2 for j in range(len(joint_velocitys))])
            rv_t = np.exp(-0.1 * distance)
            sum_reward += wv * rv_t

            # End-Effector Reward (re_t)
            distance = np.sum([self.euclidean_distance(exp_component_coordinates_world[e], component_coordinates_world[e])**2 for e in range(len(component_coordinates_world))])
            re_t = np.exp(-40 * distance)
            sum_reward += we * re_t

            # Root Pose Reward (rrp_t)
            root_position_distance = self.euclidean_distance(exp_pos, pos_array)
            root_rotation_distance = self.euclidean_distance(exp_euler_array, euler_array)
            rrp_t = np.exp(-20 * root_position_distance**2 - 10 * root_rotation_distance**2)
            sum_reward += wrp * rrp_t

            # Root Velocity Reward (rrv_t)
            root_linear_velocity_distance = self.euclidean_distance(exp_vel_array, vel_array)
            root_angular_velocity_distance = self.euclidean_distance(exp_angvel_array, angvel_array)
            rrv_t = np.exp(-2 * root_linear_velocity_distance**2 - 0.2 * root_angular_velocity_distance**2)
            sum_reward += wrv * rrv_t

        if "Modified_Peng_et_al_reward" in self.RewardsType:
            # Extract expert trajectory data
            exp_actions, exp_pos, exp_euler_array, exp_vel_array, exp_angvel_array, exp_joint_angles, exp_joint_torques, exp_joint_velocitys, exp_component_coordinates_world = expert_trajectory
            
            # Weights from the provided reward function
            wp, wv, we,   wt, wrp, wrv,   wa, wlt = 0.5, 0.05, 0.15,   0.05, 0.1, 0.05,   0.15, 0.05
            #wp, wv, we,   wt, wrp, wrv,   wa, wlt = 0.1, 0.05, 0.5,   0.05, 0.1, 0.1,   0.15, 0.05

            # Pose Reward (rp_t)
            distance = np.sum([self.euclidean_distance(exp_joint_angles[j], joint_angles[j])**2 for j in range(len(joint_angles))])
            rp_t = np.exp(-20 * distance/((2*math.pi)**2))
            sum_reward += wp * rp_t

            # Velocity Reward (rv_t)
            distance = np.sum([self.euclidean_distance(exp_joint_velocitys[j], joint_velocitys[j])**2 for j in range(len(joint_velocitys))])
            rv_t = np.exp(-0.5 * distance/((2*math.pi)**2))
            sum_reward += wv * rv_t

            # End-Effector Reward (re_t)
            distance = np.sum([self.euclidean_distance(exp_component_coordinates_world[e], component_coordinates_world[e])**2 for e in range(len(component_coordinates_world))])
            re_t = np.exp(-20 * distance)
            sum_reward += we * re_t



            # Joint_torque Reward (rt_t)
            distance = np.sum([self.euclidean_distance(exp_joint_torques[j], joint_torques[j])**2 for j in range(len(joint_torques))])
            rt_t = np.exp(-0.5 * distance)
            sum_reward += wt * rt_t

            # Root Pose Reward (rrp_t)
            root_position_distance = self.euclidean_distance(exp_pos, pos_array)
            root_rotation_distance = self.euclidean_distance(exp_euler_array, euler_array)/(2*math.pi)
            rrp_t = np.exp(-40 * root_position_distance**2 -20 * root_rotation_distance**2)
            sum_reward += wrp * rrp_t

            # Root Velocity Reward (rrv_t)
            root_linear_velocity_distance = self.euclidean_distance(exp_vel_array, vel_array)
            root_angular_velocity_distance = self.euclidean_distance(exp_angvel_array, angvel_array)/(2*math.pi)
            rrv_t = np.exp(-0.5*root_linear_velocity_distance**2 -0.5*root_angular_velocity_distance**2)
            sum_reward += wrv * rrv_t



            # Extract expert trajectory data
            # Calculate the distance between the expert action and the current action
            distance = self.euclidean_distance(exp_actions[:-1], actions)
            rwa_t = np.exp(-5 * distance)
            sum_reward += rwa_t*wa

            #give reward for using the least amount of force, Verbesserungspotential
            power_distance = np.sum([(np.linalg.norm(joint_torques[j])*joint_velocitys[j])**2 for j in range(len(joint_torques))])
            #print("power_joint:", power_joint)
            rpw_t = np.exp(-0.1 * power_distance)
            sum_reward += rpw_t*wlt


            

        #Other rewards /////////////////////////////////////////////////////////////////////////
        if "action_imitation" in self.RewardsType:
            # Extract expert trajectory data
            # Calculate the distance between the expert action and the current action
            distance = self.euclidean_distance(expert_trajectory[0][:-1], actions)
            sum_reward += np.exp(-5 * distance)*0.5

        if "stability" in self.RewardsType:
            #reward for being stable
            # Calculate reward components
            angle_x = abs(informations[1][0])+abs(informations[1][2])/2  # Angle in the y-direction
            stable_reward = 0.01**(angle_x/(math.pi*2))
            
            # Calculate total reward
            sum_reward += stable_reward*2.5
            #print("reward3:", reward3)

        if "efficiency" in self.RewardsType:
            #give reward for making the leas amount of steps/movements
            joint_vel = [joint_velocitys[i]/(2*math.pi) for i in range(self.ROS_Env.numMotors)]# jointvelocity
            #make all values in joint_vel positive
            #print("joint_vel:", joint_vel)
            self.joint_velocities.append(joint_vel)
            #calculate the average joint_velocities
            #print("joint_velocities:", self.joint_velocities)
            joint_velocities_array = np.array(self.joint_velocities)
            #average_joint_vel = np.mean(joint_velocities_array, axis=0)

            #print("average_joint_vel:", average_joint_vel)
            
            #count the amount of turningpoints that a joint has made

            #get vertical list of the joint_velocities e.g. example_array[1:x]
            all_turningpoints = 0
            transposed_joint_velocities = joint_velocities_array.T
            for joint_vel_history in transposed_joint_velocities:
                num_turningpoints = self.turningpoints(joint_vel_history)
                all_turningpoints += num_turningpoints

            reward = 0.1**(all_turningpoints/self.ROS_Env.numMotors)
            sum_reward += np.clip(reward, 0, 1)
            
        
        if "x_vel" in self.RewardsType:
            #reward for moving in the x direction
            sum_reward += np.clip(vel_array[0], -1, 1)*5.0
        
        if "life_time" in self.RewardsType:
            #increase the reward when the robot lives for longer time
            sum_reward += 5#np.clip(5**(self.ROS_Env.simulation_steps/self.maxSteps), 0, 1)

        if "vel_max_cost" in self.RewardsType:
            #reward for moving in the x direction
            tot_vel_vector = np.linalg.norm(vel_array)
            #print("tot_vel_vector:", tot_vel_vector)
            sum_reward += np.clip(1-tot_vel_vector, 0, 1)

        if "exploration" in self.RewardsType:
            #reward for moving in the x direction

            #updat the action_exploration_high and action_exploration_low
            for joint_index, joint_range in enumerate(self.ROS_Env.joint_ranges):
                self.action_exploration_high[joint_index] = max(self.action_exploration_high[joint_index], joint_angles[joint_index])
                self.action_exploration_low[joint_index] = min(self.action_exploration_low[joint_index], joint_angles[joint_index])
            #calculate the distance between the expert action and the current action
            difference_low_high = np.absolute(self.action_exploration_high - self.action_exploration_low)
            sum_reward += np.clip(1-np.mean(difference_low_high/self.max_action_difference), 0, 1)*0.2

        if "center_mass_height" in self.RewardsType:
            #reward for moving in the z direction
            sum_reward += np.clip(pos_array[2]/0.2, 0, 1)*2
            
                




        if self.render_mode=="human":
            print("reward:", sum_reward)

        #normalize the reward and clip it and return it
        return sum_reward
    
    def turningpoints(self, x):
        N=0
        for i in range(1, len(x)-1):
            if ((x[i-1] < x[i] and x[i+1] < x[i]) or (x[i-1] > x[i] and x[i+1] > x[i])):
                N += 1
        return N 

    def euclidean_distance(self, list1, list2):
        """Calculate the euclidean distance between two lists.
        return: float"""
        array1, array2 = np.array(list1), np.array(list2)
        return np.linalg.norm(array1 - array2)
    
    def check_terminated(self, informations, reward):
        #pos_array, euler_array, vel_array, angvel_array, joint_angles, joint_torques, joint_velocitys, foot_contacts, component_coordinates = informations
        #check if the robot is terminated
        #checks how many steps where simulated, stops the simulation when the animation is over
        #if (self.ROS_Env.simulation_steps*10*self.speed)+self.start_recording_time > self.end_recording_time:
        if self.Horizon_Length:
            #check if the robot is terminated
            #checks how many steps where simulated
            if abs(informations[1][0]) > math.pi*0.25 or abs(informations[1][1]) > math.pi*0.25  or informations[7][2][2] < 0.02 and self.ROS_Env.simulation_steps > 50:
                terminated = True
                #print("terminated")
            elif "Modified_Peng_et_al_reward" in self.RewardsType and (self.expert_record_steps < self.ROS_Env.simulation_steps or self.calculated_goal_dist[1] > 0.4):
                terminated = True
            elif "multi_task_goal" not in self.RewardsType and (informations[0][2] < 0.07 or informations[0][0] < -1 or abs(informations[0][1]) > 1.5):
                terminated = True
            else:
                terminated = False
        # elif self.Horizon_Length and self.ROS_Env.simulation_steps >= self.maxSteps:
        #     terminated = True
        else:
            terminated = False
        return terminated
    
    def check_truncated(self):
        """Check if the episode was truncated. That means when the limit of maximal steps in a episode is reached."""
        if self.ROS_Env.simulation_steps >= self.maxSteps:
            return True
        else:
            return False
        

    def remember_rhythm(self, action_rhythm):
        self.rhythm += action_rhythm
        #clip the rhythm
        cliped_rhythm = np.clip(self.rhythm, -1, 1)
        self.rhythm = cliped_rhythm

    
    def step(self, full_action):
        #print("step start", len(full_action))
        # Execute one time step within the environment

        if "rhythm" in self.ObservationType:
            rythm_action = full_action[-1]
            self.remember_rhythm(rythm_action)
            action = full_action[:-1]
            #print("rhythm:", self.rhythm)
        else:
            action = full_action
        
        if "Modified_Peng_et_al_reward" in self.RewardsType or "Peng_et_al_reward" in self.RewardsType or "multi_task_goal" in self.RewardsType:
            #get the nearest time to the current time
            nearest_time = [key for key in self.recorded_motions.keys() if key <= self.ROS_Env.simulation_steps][-1]
            #print("actionkey2", nearest_time)
            #print("nearest_time:", nearest_time, rest_time, "current_time", current_time)
            #get the expert action
            self.pres_expert_trajectory = self.recorded_motions[nearest_time]

            time_future = 0.3
            nearest_time = [key for key in self.recorded_motions.keys() if key <= self.ROS_Env.simulation_steps+time_future*100][-1]
            self.future_expert_trajectory = self.recorded_motions[nearest_time]
        else:
            self.pres_expert_trajectory = None
            self.future_expert_trajectory = None
        self.ROS_Env.send_action(action)

        
        #just sending the action to the real robot
        if self.real_robot and self.robot_precise_mode:
            #reads the current joint angles from the simulated robot and sends them to the real robot
            precise_action = self.ROS_Env.get_action_from_actual_joint_angles()
            print("precise_action:", precise_action)
            self.Robot_Env.send_action(precise_action)

        elif self.real_robot:
            self.Robot_Env.send_action(action)

        # Step the simulation
        self.ROS_Env.render(self.simulation_stepsize)

        if self.restriction_2D:
            self.ROS_Env.restrict_2D()






        #get the outputs
        information = self.ROS_Env.get_informations()
        observation = self.get_observation(information)
        reward = self.calculate_reward(information, actions=full_action, observation=observation, expert_trajectory=self.pres_expert_trajectory)
        terminated = self.check_terminated(information, reward)
        truncated = self.check_truncated()
        info = {}

        # Append the new observation to the stack
        if self.ROS_Env.simulation_steps % (self.obs_time_space*100/self.n_stack) == 0:
            #print("New Observation", self.ROS_Env.simulation_steps)
            self.stacked_frames.append(observation)
        else:
            #just replace the last(so the one in -1 possition) observation with the new observation
            self.stacked_frames[-1] = observation
            #print("Replace Observation", self.ROS_Env.simulation_steps)
        
        # Concatenate stacked frames
        stacked_obs = np.concatenate(self.stacked_frames, axis=None) 
        #print("step", len(stacked_obs))
        if self.gym_type:
            return stacked_obs, reward, terminated, info
        else:
            return stacked_obs, reward, terminated, truncated, info
    


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ROS_Env.simulation_steps = 0
        self.rhythm = 0


        #select a random expert
        if "Peng_et_al_reward" in self.RewardsType or "multi_task_goal" in self.RewardsType or "Modified_Peng_et_al_reward" in self.RewardsType:
            self.recorded_motions = self.recorded_motions_list[np.random.randint(0, len(self.recorded_motions_list))]
            self.pres_expert_trajectory = self.recorded_motions[0]
            self.future_expert_trajectory = self.recorded_motions[0]
            self.expert_record_steps = list(self.recorded_motions.keys())[-1]
            # self.expert_record_steps = np.random.randint(0, len(self.recorded_motions))
            #self.recorded_movement = self.recorded_motions[self.expert_record_steps]
            self.expert_coord_num = len(self.recorded_motions[0][-1])

        if "exploration" in self.RewardsType:
            #when activated the robot get more and more a higher reward for exploring new actions without dieing.
            self.action_exploration_high = np.array([])
            self.action_exploration_low = np.array([])
            self.max_action_difference = np.absolute(np.array(self.ROS_Env.joint_ranges)[:,1] - np.array(self.ROS_Env.joint_ranges)[:,0])
            for joint_range in self.ROS_Env.joint_ranges:
                middle_rad = (joint_range[0]+joint_range[1])/2
                self.action_exploration_high = np.concatenate([self.action_exploration_high, [middle_rad]])
                self.action_exploration_low = np.concatenate([self.action_exploration_low, [middle_rad]])

        if "multi_task_goal" in self.RewardsType:
            #task: follow a certain point in the room, for futher integration to use joysticks to control the robot
            self.direction_vec = np.array([self.goal_min_max_pace[1], 0, 0])
            self.goal_pos_vec = np.array([0, 0, 0.1])

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        self.ROS_Env.reset(seed=seed, options=options)
        if self.real_robot:
            self.Robot_Env.reset(seed=seed, options=options)
        
        # Get initial observation
        information = self.ROS_Env.get_informations()
        observation = self.get_observation(information)
        
        # Clear the frame stack and add the initial observation
        self.stacked_frames.clear()
        for _ in range(self.n_stack):
            self.stacked_frames.append(observation)
        # Return the stacked frames as the initial observation
        stacked_obs = np.concatenate(self.stacked_frames, axis=None)

        # put noise on the observation
        observation = np.clip(stacked_obs + np.random.uniform(noise_low, noise_high, size=stacked_obs.shape), -1, 1)
        observation[-1] = self.rhythm #because we don't want the rhythm to be changed by the noise
        
        #print("restart:",len(stacked_obs))
        if self.gym_type:
            return stacked_obs, {}
        else:
            return stacked_obs, {}
        
    def close(self):
        self.ROS_Env.close()
        if self.real_robot:
            self.Robot_Env.close()
        super().close()


if __name__ == "__main__":
    env = CustomEnv(ModelType="SAC", gui=True, render_mode="fast", rewards_type=["Modified_Peng_et_al_reward"], observation_type=["joint_forces", "joint_angles", "goal_orientation"], simulation_stepSize=5, restriction_2D=True)
    env.reset()

    # ['Revolute 38', 'Revolute 40', 'Rigid 50', 'Revolute 51', 'Revolute 52', 'Revolute 53', 'Revolute 54', 
    #  'Revolute 55', 'Revolute 56', 'Revolute 39', 'Revolute 41', 'Rigid 42', 'Rigid 43', 'Revolute 48', 'Revolute 49', 'Rigid 46', 'Rigid 47']

    for i in range(5000):
        # for i in range(70):
        #     env.step([6 for i in range(num_motors)]+[1])
        #returning = env.step(np.array([0 for i in range(9)]))
        env.step([-0.4, 0,   -0.5, 0.3,  -0.5, 0.3,    0.2, 0.2])
            #print("returning:", returning[0])
        
        # for i in range(5):
        #     env.step([0,0]+[3,3,3]+[3,3,3]+[3,6]+[0])

        # for i in range(50):
        #     env.step([5,6]+[3,8,7]+[3,8,7]+[4,4]+[1])
        #     #env.step([8 for i in range(num_motors)]+[1])

        # for i in range(50):
        #     env.step([5,6]+[6,5,5]+[6,5,5]+[5,5]+[1])

    env.close()