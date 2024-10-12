import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

#import matplotlib.pyplot as plt

import numpy as np
import time
import math
import json
import os
import sys

# Assuming Bunny_Project_v2 is the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

from envs.Servo_Controler_v4 import *




class Robot_env():
    """
    This is a pybullet environment for the bunny robot. The robot has 12 motors. But only 8 of them can be controlled via the action space.
    The action space is a 8 dimensional vector. The first two values are for the spine, the next four for the legs and the last two for the arms.

    """
    
    def __init__(self,
                 gui=True,
                 simulation_speed="human",
                 status_types=["position", "orientation", "linear_velocity", "angular_velocity", "joint_angles", "joint_torques", "joint_velocitys", "foot_contacts", "component_coordinates_world", "component_coordinates_local"],
                 ):
        super().__init__()
        self.GUI = gui

        self.MAXFORCE = 2.9#2.941995#5#in Newton   3 N/m
        self.MAXVELOCITY = 4.65#(2*math.pi)/(0.222*6)#in rad/s

        #define the joint ranges
        self.joint_ranges = [(-0.45, 0.2), (-0.523599, 0.523599), (-2.617994, 0.610865), (-1.22173, 1.134464), (-1.047198, 0.785398), (-2.617994, 0.610865), (-1.22173, 1.134464), (-1.134464, 0.785398), (-0.2, 0.45), (-0.523599, 0.523599), (-0.872665, 0.872665), (-0.872665, 0.872665)]
        #little adjustment for jointpositions
        self.joint_action_adjustments = [0, 0,    0.1, 0.1,    0,0,  0,0]


        self.simulation_speed = simulation_speed
        self.simulation_steps = 0
        self.last_render_time = time.time()

        #region Robot
        self.RealBunny_Controler = ServoControler()

        # make a custom function that always returns the preferred states of the robot
        self.get_informations = self.make_get_informations(status_types)


    def get_link_infos(self):
        pass


    def make_get_informations(self, status_types):
        """Create a function that returns the preferred states of the robot
        status_types: list of strings, the types of the status that should be returned
        return: function

        This allows us to save resources, like RAM, because we can only get the informations we need.
        """
        #create a list of all the status types
        # lambda_list = []

        # if "position" in status_types:
        #     lambda_list.append(lambda: p.getBasePositionAndOrientation(self.robot)[0])
        # if "orientation" in status_types:
        #     lambda_list.append(lambda: p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1]))
        # if "linear_velocity" in status_types:
        #     lambda_list.append(lambda: p.getBaseVelocity(self.robot)[0])
        # if "angular_velocity" in status_types:
        #     lambda_list.append(lambda: p.getBaseVelocity(self.robot)[1])
        # if "joint_angles" in status_types:
        #     lambda_list.append(lambda: [p.getJointState(self.robot, i)[0] for i in self.Motors_index])
        # if "joint_torques" in status_types:
        #     lambda_list.append(lambda: [p.getJointState(self.robot, i)[2] for i in self.Motors_index])
        # if "joint_velocitys" in status_types:
        #     lambda_list.append(lambda: [p.getJointState(self.robot, i)[1] for i in self.Motors_index])
        # if "foot_contacts" in status_types:
        #     foot_bodys = [5, 8, 13, 14]
        #     foot_contacts = []
        #     for foot in foot_bodys:
        #         foot_contacts.append(p.getContactPoints(self.robot, self.ground, foot))
        #     lambda_list.append(lambda: foot_contacts)
        # if "component_coordinates_world" in status_types:
        #     lambda_list.append(lambda: [p.getLinkState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))])
        # if "component_coordinates_local" in status_types:
        #     lambda_list.append(lambda: [p.getLinkState(self.robot, i)[2] for i in range(p.getNumJoints(self.robot))])

        # def get_informations():
        #     return [func() for func in lambda_list]

        # return get_informations
        pass


    def euclidean_distance(self, list1, list2):
        """Calculate the euclidean distance between two lists.
        return: float"""
        array1, array2 = np.array(list1), np.array(list2)
        return np.linalg.norm(array1 - array2)
    
    def add_linked_joints_to_actions(self, ordered_joints):
        """Adds the joints for the foot, that are linked with the lower leg
        """
        spine_actions = ordered_joints[0:2]
        leg_actions_right = ordered_joints[2:4]
        leg_actions_left = ordered_joints[4:6]
        arm_actions = ordered_joints[6:8]

        #add the linked joints
        foot_joint_right = [-leg_actions_right[1]-0.1]
        foot_joint_left = [-leg_actions_left[1]-0.1]

        return np.concatenate([spine_actions, leg_actions_right, foot_joint_right, leg_actions_left, foot_joint_left, arm_actions])
    
    def _map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    
    def convert_to_robo_jointOrder(self, ordered_joints, symetric=-1):
        """Das ordentliche wird in wirrwar gebracht. Damit man dises dann Pybullet übergeben kann.
            Fortmat of the parameter ordered_joints: [0,0](SpineMotors)+[3,3,3](LegLeft)+[3,3,3](LegRight)+[3,6](ArmLeftRight)
        
        """
        wirrwar = [0 for i in range(12)]
        
        wirrwar[0] = ordered_joints[0] 
        wirrwar[1] = ordered_joints[1]
        wirrwar[2] = symetric*ordered_joints[5]
        wirrwar[3] = symetric*ordered_joints[6]
        wirrwar[4] = symetric*ordered_joints[7]
        wirrwar[5] = symetric*ordered_joints[2]
        wirrwar[6] = ordered_joints[3]
        wirrwar[7] = symetric*ordered_joints[4]
        wirrwar[8] = symetric*ordered_joints[0]
        wirrwar[9] = symetric*ordered_joints[1]
        wirrwar[10] = ordered_joints[9]
        wirrwar[11] = ordered_joints[8]
        return wirrwar
    
    def InverseKinematics(self, Knee_angle, FrontHip_angle, degree=False):
        """
        give the angle of the Knee and FrontHip.
        Returns: the angle of the BackHip = b1. b1 goes counted anticlockwise from 0 to 2*pi
        """
        pi = math.pi
        #Motor 1 Position
        Motor1_coo = np.array([0, 0])

        #set some constants
        S1 = 0.07#FrontHip to Knee
        S4 = 0.042#Knee to MiddleUnderLeg
        t2 = np.array([0.03, 0])#Distance between the FrontHip and the BackHip
        S2 = 0.04#BackHip to Knee2
        S3 = 0.07#Knee2 to MiddleUnderLeg

        S5 = 0.03#MiddleUnderLeg to Foot
        FrontHip_angle = -FrontHip_angle
        Knee_angle = -Knee_angle
        if degree:
            FrontHip_angle = FrontHip_angle/180*math.pi
            Knee_angle = Knee_angle/180*math.pi
        
        a1 = self._map(FrontHip_angle, -math.pi, math.pi, 0, 2*math.pi)
        a2 = 0.5*math.pi-Knee_angle

        print("a1:", a1/pi*180, "a2:", a2/pi*180, "Knee_angle:", Knee_angle/pi*180, "FrontHip_angle:", FrontHip_angle/pi*180)


        #calculate the angle of the BackHip
        # m1 = np.array([math.cos(-a1+pi), math.sin(-a1+pi)])*S1
        # m4 = np.array([math.cos(-a1+pi*1.5-a2), math.sin(-a2+pi*1.5-a2)])*S4
        # m5 = np.array([math.cos(-a1+pi*1.5-a2), math.sin(-a2+pi*1.5-a2)])*S5
        m1 = np.array([math.cos(a1), math.sin(a1)])*S1
        m4 = np.array([math.cos(a1+a2), math.sin(a1+a2)])*S4
        m5 = np.array([math.cos(a1+a2), math.sin(a1+a2)])*S5


        #show_arms(m1, (0,0), (0, 0), m4, m5, t2)
        P = m1+m4+m5
        T = m1+m4
        t1_vec =  T-t2
        t1 = np.linalg.norm(t1_vec)

        z = (S2**2-S3**2+t1**2)/(2*S2*t1)
        # shoulden't be greater than 1
        if z > 1:
            z = 1
            print("z is greater than 1")
        elif z < -1:
            z = -1
            print("z is smaller than -1")

        phi1 = math.acos(z)
        
        #atan doesn't give angels above 90 degrees, so I have to calculate with sin and cos
        #phi2 = math.asin(t1_vec[1]/(t1_vec[0]**2+t1_vec[1]**2)**0.5)
        if t1_vec[0] > 0:
            phi2 = math.atan(t1_vec[1]/t1_vec[0])
            phi2 = pi+phi2
        elif t1_vec[0] < 0:
            phi2 = math.atan(t1_vec[1]/t1_vec[0])
        else:
            phi2 = pi/2

        #phi2 = math.atan2(t1_vec[1], t1_vec[0])

        
        b1 = phi1+phi2
        m2 = np.array([math.cos(b1+pi)*S2, math.sin(b1+pi)*S2])
        m3 = t1_vec-m2
        print("phi1: ", phi1, "phi2: ", phi2, t1_vec, "b1", b1/pi*180)


        return b1
    
    def get_angle_in_range(self, angle_percent, joint_index):
        #get the joint range
        joint_range = self.joint_ranges[joint_index]
        #calculate the angle
        return joint_range[0] + (joint_range[1] - joint_range[0]) * (angle_percent + 1) / 2
    
    def convert_to_real_Bunny_jointOrder(self, ordered_joints, symetric=1):
        """Das ordentliche wird in wirrwar gebracht. Damit man dises dann Pybullet übergeben kann.
            Fortmat of the parameter ordered_joints: [0,0](SpineMotors)+[3,3,3](LegLeft)+[3,3,3](LegRight)+[3,6](ArmLeftRight)
        
        """
        real_jointOrder = [0 for i in range(9)]
        
        #the spine joints

        real_jointOrder[1] = np.clip(ordered_joints[1]*-90+ordered_joints[0]*-90, -100, 35)
        real_jointOrder[2] = np.clip(ordered_joints[1]*90+ordered_joints[0]*-90, -100, 35)


        real_jointOrder[4] = self._map(ordered_joints[2], 1, -1, self.joint_ranges[5][0], self.joint_ranges[5][1])/(2*math.pi)*360
        real_jointOrder[6] = 180 - self.InverseKinematics(self.get_angle_in_range(ordered_joints[3], 6), real_jointOrder[4]/360*2*math.pi)/(2*math.pi)*360
        #real_jointOrder[7] = 0#ordered_joints[4]

        real_jointOrder[3] = self._map(ordered_joints[5], 1, -1, self.joint_ranges[2][0], self.joint_ranges[2][1])/(2*math.pi)*360
        real_jointOrder[5] = 180 - self.InverseKinematics(self.get_angle_in_range(ordered_joints[6], 3), real_jointOrder[3]/360*2*math.pi)/(2*math.pi)*360
        #real_jointOrder[4] = 0#ordered_joints[7]


        real_jointOrder[8] = self._map(ordered_joints[8], 1, -1, self.joint_ranges[11][0], self.joint_ranges[11][1])/(2*math.pi)*360
        real_jointOrder[7] = self._map(ordered_joints[9], 1, -1, self.joint_ranges[10][0], self.joint_ranges[10][1])/(2*math.pi)*360

        #because the servonumbers are in range 1 to 8, the first 0. index have to be deleted
        real_jointOrder = real_jointOrder[1:]
        #print("real_jointOrder:", real_jointOrder)

        return real_jointOrder
    
    def convert_actionPercent_to_radiant_of_jointrange(self, robotMotorOrder_actions):
        angle_list = []
        for joint_index, targetAngle in enumerate(robotMotorOrder_actions):
            #get the joint range
            joint_range = self.joint_ranges[joint_index]
            #calculate the angle
            angle = joint_range[0] + (joint_range[1] - joint_range[0]) * (targetAngle + 1) / 2
            angle_list.append(angle)
        return angle_list

    
    def send_action(self, action):
        """Just sends the action. But nothing will be rendered"""
        #add the joint_action_adjustments
        action = np.array(action) + np.array(self.joint_action_adjustments)
        action = self.add_linked_joints_to_actions(action)

        #send commands action to the robot
        action_for_all_Motorjoints = self.convert_to_real_Bunny_jointOrder(action)

        self.RealBunny_Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], action_for_all_Motorjoints, 4000, 255)
        

    def restrict_2D(self, correction_strength=20):
        pass


    def reset(self, seed=None, options=None):
        self.simulation_steps = 0
        self.last_render_time = time.time()

        # Reset the seros position to the start position
        self.send_action([0, 0, 0, 0, 0, 0, 0, 0])

        


    def back_to_start(self):
        # Reset the position of the robot
        pass

    def render(self, step_num = 1):
        """Render the environment to the screen
        step_num: int, number of steps to simulate. One step is 0.01 seconds
        """
        # Render the environment to the screen
        # Step the simulation
        #default fixedTimeStep = 0.00416 = 1/240 = 240Hz
        self.simulation_steps += step_num

        # Step the simulation for step_num*0.01 seconds
        # 5*0.01/(1/240)

        #Time to collect servo data............................

        if self.simulation_speed == "human":
            #print("simulation_steps:", self.last_render_time)
            elapsed_time = time.time() - self.last_render_time
            remaining_time = max(0.01*step_num - elapsed_time, 0)
            #print(f"elapsed_time: {elapsed_time:.4f} seconds, {remaining_time:.4f} seconds remaining")
            time.sleep(remaining_time)
            self.last_render_time = time.time()
        else:
            pass

    def close(self):
        # Close the environment
        print("Environment closed")


if __name__ == "__main__":
    env = Robot_env()
    env.render(5)
    env.get_link_infos()
    env.render()
    print("Simulation started")

    env.send_action([0, 0, 0, 0, 0, 0, 0, 0])

    while True:
        time.sleep(0.5)
        pass

    for i in range(1000):
        env.render()
        #get the informations
        pos_array, euler_array, vel_array, angvel_array, all_joint_states, foot_contacts, body_positions, joint_coordinates = env.get_informations()
        #give the robot a random action
        action = np.random.uniform(-1, 1, 12)
        env.send_action(action)


    env.close()