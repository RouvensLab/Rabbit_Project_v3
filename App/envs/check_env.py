import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque

import pybullet as p
import pybullet_data

#import matplotlib.pyplot as plt

import numpy as np
import time
import math
import json




class ROS_env():
    """
    This is a pybullet environment for the bunny robot. The robot has 12 motors. But only 8 of them can be controlled via the action space.
    The action space is a 8 dimensional vector. The first two values are for the spine, the next four for the legs and the last two for the arms.

    """
    
    def __init__(self,
                 gui=True,
                 simulation_speed="human",
                 status_types=["position", "orientation", "linear_velocity", "angular_velocity", "joint_angles", "joint_torques", "joint_velocitys", "foot_contacts", "component_coordinates_world", "component_coordinates_local"],
                 terrain_type = "random_terrain",
                 hung = False,
                 change_body_params = None
                 ):
        super().__init__()
        self.GUI = gui
        self.hung = hung
        
        self.MAXFORCE = 2.9#2.941995#5#in Newton   3 N/m
        self.MAXVELOCITY = 4.65#(2*math.pi)/(0.222*6)#in rad/s


        #initialization of pybullet
        if gui:
            connection_id = p.connect(p.GUI)            #RL 2024
        else:
            connection_id = p.connect(p.DIRECT)         #RL 2024
        print(f"ByBullet started with connection ID : {connection_id}") #RL 2024


        self.simulation_speed = simulation_speed
        self.simulation_steps = 0
        self.last_render_time = time.time()



        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(p.isConnected())
        self.terrain_type = terrain_type
        if terrain_type == "uneven_terrain":
            # Create heightfield data: a grid of random heights to simulate uneven terrain
            num_rows = 256  # Number of rows in the heightfield grid
            num_columns = 256  # Number of columns in the heightfield grid
            height_scale = 0.1  # Scale the height values to control the roughness of the terrain

            # Generate random heights for the terrain
            heightfield_data = np.random.uniform(-1, 1, size=(num_rows, num_columns))
            heightfield_data = heightfield_data * height_scale

            # Flatten the heightfield data into a 1D array, as required by PyBullet
            heightfield_data_flattened = heightfield_data.flatten()

            # Create heightfield collision shape
            # random meshScale, range from 0.05 to 0.5
            rand_meshScale = [0.1, 0.1, 0.1]#np.random.uniform(0.05, 1, 3)


            terrain_shape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                meshScale=rand_meshScale,  # Scale of the heightfield
                                                heightfieldTextureScaling=(num_rows-1)/2,
                                                heightfieldData=heightfield_data_flattened,
                                                numHeightfieldRows=num_rows,
                                                numHeightfieldColumns=num_columns)

            # Create the terrain in the simulation
            self.ground = p.createMultiBody(0, terrain_shape)

            # Optionally, you can set additional visual or physical properties for the terrain
            p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1])


        elif self.terrain_type == "random_terrain":            
            # Create heightfield data: a grid of random heights to simulate uneven terrain
            num_rows = 256  # Number of rows in the heightfield grid
            num_columns = 256  # Number of columns in the heightfield grid
            height_scale = 0.2  # Adjust this for terrain roughness (forest ground shouldn't be too extreme)
            
            # Generate random heights for the terrain (use smoother ranges for natural feel)
            heightfield_data = np.random.uniform(-0.5, 0.5, size=(num_rows, num_columns))
            heightfield_data = heightfield_data * height_scale
            
            # Flatten the heightfield data into a 1D array, as required by PyBullet
            heightfield_data_flattened = heightfield_data.flatten()
            
            # Create heightfield collision shape
                # Create heightfield collision shape with varied meshScale for more natural terrain
            meshScale = [
                np.random.uniform(0.08, 0.22),  # Scale for x-axis (small variation)
                np.random.uniform(0.08, 0.22),  # Scale for y-axis (small variation)
                np.random.uniform(0.1, 0.35)    # Larger variation for z-axis to simulate bumps and dips
            ]
            
            terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=meshScale,  # Scale of the heightfield
                heightfieldTextureScaling=(num_rows-1)/2,
                heightfieldData=heightfield_data_flattened,
                numHeightfieldRows=num_rows,
                numHeightfieldColumns=num_columns
            )
            
            # Create the terrain in the simulation
            self.ground = p.createMultiBody(0, terrain_shape)
            
            # Optionally, set additional visual or physical properties for the terrain
            p.resetBasePositionAndOrientation(self.ground, [0, 0, 0], [0, 0, 0, 1])

            # You could also apply textures or additional friction properties if needed
        else:
            self.ground = p.loadURDF("plane.urdf")


        self.robotpos = [0, 0, 0.2]  #meters

        self.robot = p.loadURDF(r"URDF_Files\_Compleat_Bunny_replica_v2_light_v5\urdf\_Compleat_Bunny_replica_v2.xacro", self.robotpos[0], self.robotpos[1], self.robotpos[2])
        #turn the robot around
        p.resetBasePositionAndOrientation(self.robot, self.robotpos, [0, 0, 1, 1])



        p.setGravity(0, 0, -9.81)
        #set the grounds lateral_friction to 1
        p.changeDynamics(self.ground, -1, lateralFriction=1)
        #set the robots lateral_friction to 1
        print("lateralFriction of Robot: ", p.getDynamicsInfo(self.robot, -1)[1])
        p.changeDynamics(self.robot, -1, 
                         lateralFriction=1, 
                         jointDamping=0.1  # Damping for compliance
                         )

        #the frequency of the simulation
        print("PhysicsEngineParameter: ",p.getPhysicsEngineParameters())




        #get the names of the joints
        self.joint_names = [p.getJointInfo(self.robot, i)[1].decode("utf-8") for i in range(p.getNumJoints(self.robot))]
        print("joint_names:", self.joint_names)
        
        #get all the motors of the robot, ordered in the way they are in the URDF file. All indexes of the motors == [0, 9, 1, 10, 6, 7,8, 3, 4, 5, 14, 13]
        self.Motors_index = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]
        self.numMotors = len(self.Motors_index)
        #when there are other physical properties for the motors, they can be set here
        # self.Motors_strength = [self.MAXFORCE*5, self.MAXFORCE*5, self.MAXFORCE, self.MAXFORCE, self.MAXFORCE, self.MAXFORCE,self.MAXFORCE,self.MAXFORCE,self.MAXFORCE*5 ,self.MAXFORCE*5, self.MAXFORCE, self.MAXFORCE]
        # self.Motors_velocity = [self.MAXVELOCITY/5, self.MAXVELOCITY/5, self.MAXVELOCITY, self.MAXVELOCITY, self.MAXVELOCITY, self.MAXVELOCITY,self.MAXVELOCITY,self.MAXVELOCITY,self.MAXVELOCITY/5,self.MAXVELOCITY/5, self.MAXVELOCITY, self.MAXVELOCITY]
        self.Motors_strength = [self.MAXFORCE for i in range(self.numMotors)]
        self.Motors_velocity = [self.MAXVELOCITY for i in range(self.numMotors)]
        
        #get the range of the motors
        self.joint_ranges = [p.getJointInfo(self.robot, i)[8:10] for i in self.Motors_index]
        print("joint_ranges:", self.joint_ranges)
        self.numJoints = p.getNumJoints(self.robot)


        #get link name of robot
        # linkNames = [p.getJointInfo(self.robot, i)[12].decode("utf-8") for i in range(p.getNumJoints(self.robot))]
        # print("link_name:",linkNames)


         # #enableJointForceTorqueSensor
        for joint in self.Motors_index:
            p.enableJointForceTorqueSensor(self.robot, joint, enableSensor=True)

        #change physic parameters as the user wants
        if change_body_params is not None:
            for key, value in change_body_params.items():
                if "Motors_strength" in key:
                    self.Motors_strength = value
                elif "Motors_velocity" in key:
                    self.Motors_velocity = value

        # make a custom function that always returns the preferred states of the robot
        self.get_informations = self.make_get_informations(status_types)


    def get_link_infos(self):
        link_infos = [p.getLinkState(self.robot, i) for i in range(p.getNumJoints(self.robot))]
        #get all the names of the links
        link_names = [p.getJointInfo(self.robot, i)[12].decode("utf-8") for i in range(p.getNumJoints(self.robot))]
        #get all the weights of the links
        link_masses = [p.getDynamicsInfo(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))]
        #get the total mass of the robot
        total_mass = sum(link_masses)
        print("total_mass:", total_mass, "\n")

        print("link_masses:", link_masses, "\n")

        print("link_names:", link_names, "\n")

        print("link_infos:", link_infos, "\n")

        #_show_weights_in_UI
        link_positions = [link_info[0] for link_info in link_infos]
        p.addUserDebugPoints(link_positions, [[0, 1, 0] for i in range(len(link_positions))], 5)
        for i, link_mass in enumerate(link_masses):
            #make a dot at the position of the center mass of the link
            p.addUserDebugText(f"Nr:{i}, {link_names[i]} m={str(round(link_mass, 3))}", link_infos[i][0], [1, 0, 0], 1)
        
        #show total mass at the top of the robot
        p.addUserDebugText(str(round(total_mass, 3)), [0, 0, 0.2], [1, 0, 0], 1)

    def show_world_coordinate_system(self):
        #show the world coordinate system
        p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], 1)
        p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], 1)
        p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], 1)
        #show labled axes of the coordinate system
        p.addUserDebugText("X", [1, 0, 0], [1, 0, 0], 1)
        p.addUserDebugText("Y", [0, 1, 0], [0, 1, 0], 1)
        p.addUserDebugText("Z", [0, 0, 1], [0, 0, 1], 1)

        #make dots in distances of 0.1 meters on all axes
        for i in range(1, 11):
            p.addUserDebugText(str(i), [i*0.1, 0, 0], [1, 0, 0], 1)
            p.addUserDebugText(str(i), [0, i*0.1, 0], [0, 1, 0], 1)
            p.addUserDebugText(str(i), [0, 0, i*0.1], [0, 0, 1], 1)

    def show_linked_vectors(self, vector_list):
        vector_chain = np.array(vector_list[0])
        for i in range(1, len(vector_list)):
            #print(vector_chain, vector_list[i])
            p.addUserDebugLine(vector_chain, vector_chain+vector_list[i], [1, 0, 0], 2)
            vector_chain += np.array(vector_list[i])

    def show_TextPoint(self, point_coord, text):
        #show the text at a specific point
        p.addUserDebugText(text, point_coord, [0, 0, 1], 1.2)

    def make_get_informations(self, status_types):
        """Create a function that returns the preferred states of the robot
        status_types: list of strings, the types of the status that should be returned
        return: function

        This allows us to save resources, like RAM, because we can only get the informations we need.
        """
        #create a list of all the status types
        lambda_list = []

        if "position" in status_types:
            lambda_list.append(lambda: p.getBasePositionAndOrientation(self.robot)[0])
        if "orientation" in status_types:
            lambda_list.append(lambda: p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot)[1]))
        if "linear_velocity" in status_types:
            lambda_list.append(lambda: p.getBaseVelocity(self.robot)[0])
        if "angular_velocity" in status_types:
            lambda_list.append(lambda: p.getBaseVelocity(self.robot)[1])
        if "joint_angles" in status_types:
            lambda_list.append(lambda: [p.getJointState(self.robot, i)[0] for i in self.Motors_index])
        if "joint_torques" in status_types:
            lambda_list.append(lambda: [p.getJointState(self.robot, i)[2] for i in self.Motors_index])
        if "joint_velocitys" in status_types:
            lambda_list.append(lambda: [p.getJointState(self.robot, i)[1] for i in self.Motors_index])
        if "foot_contacts" in status_types:
            foot_bodys = [5, 8, 13, 14]
            foot_contacts = []
            for foot in foot_bodys:
                foot_contacts.append(p.getContactPoints(self.robot, self.ground, foot))
            lambda_list.append(lambda: foot_contacts)
        if "component_coordinates_world" in status_types:
            lambda_list.append(lambda: [p.getLinkState(self.robot, i)[0] for i in range(p.getNumJoints(self.robot))])
        if "component_coordinates_local" in status_types:
            lambda_list.append(lambda: [p.getLinkState(self.robot, i)[2] for i in range(p.getNumJoints(self.robot))])

        if "vision" in status_types:
            lambda_list.append(lambda: p.getCameraImage(320, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

        def get_informations():
            return [func() for func in lambda_list]

        return get_informations
    

    
    def show_link_index_pos_in_sim(self, body_positions, pos_array):
        #clear the screen
        p.removeAllUserDebugItems()
        #show which link_index is which link
        for link_index, link_pos in body_positions.items():
            print(link_index, p.getJointInfo(self.robot, link_index)[12].decode("utf-8"))
            #plot a imaginary ball at a specific position
            p.addUserDebugText(str(link_index), link_pos, [1, 0, 0], 1)
        #show position of the main body
        p.addUserDebugText("MainBody", pos_array, [1, 0, 0], 1)

    def show_expert_animation(self, joint_coordinates):
        #show the experts joint_coordinates
        for i, coord in enumerate(joint_coordinates):
            p.addUserDebugText(str(i), coord, [1, 0, 0], 1)
        # green_points_list = [(0, 1, 0) for i in range(len(joint_coordinates))]
        # p.addUserDebugPoints(coord, green_points_list) 

    def euclidean_distance(self, list1, list2):
        """Calculate the euclidean distance between two lists.
        return: float"""
        array1, array2 = np.array(list1), np.array(list2)
        return np.linalg.norm(array1 - array2)
    
    def add_linked_joints_to_actions(self, ordered_joints, other_way=False):
        """Adds the joints for the foot, that are linked with the lower leg
        """
        if not other_way:
            spine_actions = ordered_joints[0:2]
            leg_actions_right = ordered_joints[2:4]
            leg_actions_left = ordered_joints[4:6]
            arm_actions = ordered_joints[6:8]

            #add the linked joints
            foot_joint_right = [-leg_actions_right[1]-0.1]
            foot_joint_left = [-leg_actions_left[1]-0.1]

            return np.concatenate([spine_actions, leg_actions_right, foot_joint_right, leg_actions_left, foot_joint_left, arm_actions])
        else:
            spine_actions = ordered_joints[0:2]
            leg_actions_right = ordered_joints[2:4]
            leg_actions_left = ordered_joints[5:7]
            arm_actions = ordered_joints[8:10]

            return np.concatenate([spine_actions, leg_actions_right, leg_actions_left, arm_actions])
        
    
    def convert_to_robo_jointOrder(self, ordered_joints, symetric=-1, other_way=False):
        """Das ordentliche wird in wirrwar gebracht. Damit man dises dann Pybullet übergeben kann.
            Fortmat of the parameter ordered_joints: [0,0](SpineMotors)+[3,3,3](LegLeft)+[3,3,3](LegRight)+[3,6](ArmLeftRight)
        
        """
        if not other_way:
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
        else:
            wirrwar = [0 for i in range(10)]

            wirrwar[0] = ordered_joints[0]
            wirrwar[1] = ordered_joints[1]

            wirrwar[2] = symetric*ordered_joints[5]
            wirrwar[3] = ordered_joints[6]
            wirrwar[4] = ordered_joints[7]

            wirrwar[5] = symetric*ordered_joints[2]
            wirrwar[6] = symetric*ordered_joints[3]
            wirrwar[7] = ordered_joints[4]

            wirrwar[8] = ordered_joints[11]
            wirrwar[9] = ordered_joints[10]

        return wirrwar
    
    def convert_actionPercent_to_radiant_of_jointrange(self, robotMotorOrder_actions):
        """
        Convert the action to the joint angles
        robotMotorOrder_actions: list of floats with the actions in the robotMotorOrder
        return: list of floats with the angles in radiant of the joints in the robotMotorOrder
        """
        angle_list = []
        for joint_index, targetAngle in enumerate(robotMotorOrder_actions):
            #get the joint range
            joint_range = self.joint_ranges[joint_index]
            #calculate the angle
            angle = joint_range[0] + (joint_range[1] - joint_range[0]) * (targetAngle + 1) / 2
            angle_list.append(angle)
        return angle_list
    
    #inverse calculation: to get the action from the joint angles
    def convert_radiant_of_jointrange_to_actionPercent(self, robotMotorOrder_angles):
        """
        Converts the joint angles to the action
        robotMotorOrder_angles: list of floats with the angles in radiant of the joints in the robotMotorOrder
        return: list of floats with the actions in the robotMotorOrder
        """
        action_list = []
        for joint_index, angle in enumerate(robotMotorOrder_angles):
            #get the joint range
            joint_range = self.joint_ranges[joint_index]
            #calculate the action
            action = 2 * (angle - joint_range[0]) / (joint_range[1] - joint_range[0]) - 1
            action_list.append(action)

        return action_list
    def get_action_from_actual_joint_angles(self):
        joint_angles = [p.getJointState(self.robot, i)[0] for i in self.Motors_index]
        #get the action from the joint angles
        action = self.convert_radiant_of_jointrange_to_actionPercent(joint_angles)
        action = self.convert_to_robo_jointOrder(action, other_way=True)
        #add the linked joints
        action = self.add_linked_joints_to_actions(action, other_way=True)
        #clip the action
        action = np.clip(action, -1, 1)
        return action

    
    def send_action(self, action):
        """Just sends the action. But nothing will be rendered"""
        
        action_for_all_Motorjoints = self.convert_to_robo_jointOrder(self.add_linked_joints_to_actions(action), symetric=-1)

        robo_joint_radiants = self.convert_actionPercent_to_radiant_of_jointrange(action_for_all_Motorjoints)

        for joint_Motor_index, targetAngle in enumerate(robo_joint_radiants):
            p.setJointMotorControl2(self.robot, self.Motors_index[joint_Motor_index], p.POSITION_CONTROL, targetAngle, force=self.Motors_strength[joint_Motor_index], maxVelocity=self.Motors_velocity[joint_Motor_index])
        

    def restrict_2D(self, correction_strength=20):
        # Get the robot's current position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        
        # Corrective force to keep the robot near y = 0
        y_error = -pos[1]  # Calculate deviation from y = 0

        
        # Apply a force in the y-direction to correct the position
        p.applyExternalForce(
            objectUniqueId=self.robot,
            linkIndex=-1,  # Apply to the base link (whole body)
            forceObj=[0, y_error * correction_strength, 0],  # Force direction and magnitude
            posObj=(0.3,0,-0.1),#np.array(pos)-np.array([0,0,0.05]),  # Point of application (center of mass)
            flags=p.LINK_FRAME  # Apply in the world frame
        )


    def reset(self, seed=None, options=None):
        self.simulation_steps = 0
        
        # # Reset the PyBullet simulation
        # p.resetSimulation()
        # p.setGravity(0, 0, -9.81)
        hight = 0.15

        angle = 0
        if seed is not None:
            np.random.seed(seed)
            #get a angle between -0.5pi and 0.5pi
            angle = np.random.uniform(-0.2*math.pi, 0.2*math.pi)

        # Reset the position of the robot
        p.resetBasePositionAndOrientation(self.robot, [0,0,hight], [0, 0, angle, 1])

        #repostion the joints
        for joint in self.Motors_index:
            p.resetJointState(self.robot, joint, 0)
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL, 0, 
                                    force=self.MAXFORCE, 
                                    maxVelocity=self.MAXVELOCITY,
                                    # positionGain=0.3,
                                    # velocityGain=0.5                                    
                                    )
            
        


    def back_to_start(self):
        # Reset the position of the robot
        p.resetBasePositionAndOrientation(self.robot, self.robotpos, [0, 0, 0, 1])

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
        for _ in range(int(step_num*0.01/(1/240))):
            if self.hung:
                #levetate the robot
                p.resetBasePositionAndOrientation(self.robot, [0,0,1], [0, 0, 0, 1])
            p.stepSimulation()

        if self.simulation_speed == "human":
            #print("simulation_steps:", self.last_render_time)
            elapsed_time = time.time() - self.last_render_time
            remaining_time = max(0.01*step_num - elapsed_time, 0)
            #print(f"elapsed_time: {elapsed_time:.4f} seconds, {remaining_time:.4f} seconds remaining")
            time.sleep(remaining_time)
            self.last_render_time = time.time()
        else:
            pass

        if self.GUI:
            p.removeAllUserDebugItems()

    def close(self):
        # Close the environment
        p.disconnect()
        print("Environment closed")


if __name__ == "__main__":
    env = ROS_env()
    env.render(5)
    env.get_link_infos()
    env.show_world_coordinate_system()
    env.render()
    print("Simulation started")


    while True:
        time.sleep(0.5)
        env.get_link_infos()
        pass

    for i in range(1000):
        env.render()
        #get the informations
        pos_array, euler_array, vel_array, angvel_array, all_joint_states, foot_contacts, body_positions, joint_coordinates = env.get_informations()
        #give the robot a random action
        action = np.random.uniform(-1, 1, 12)
        env.send_action(action)


    env.close()