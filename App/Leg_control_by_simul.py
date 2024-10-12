import math
import pygame
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from envs.Servo_Controler_v4 import *  # Import your ServoControler class

# Create an instance of the ServoControler class
#ServoManager = ServoControler()

# Function to map joystick values to servo positions
def map_joystick_to_servo(value, min_value, max_value):
    # Map joystick value (ranging from -1 to 1) to servo position (between min and max values)
    return int((value + 1) / 2 * (max_value - min_value) + min_value)

class WholeRobot:
    def __init__(self, koordinate_m1, koordinate_m2, length1, length2, length3, length4):
        self.koordinate_m1 = koordinate_m1
        self.koordinate_m2 = koordinate_m2
        self.l1 = length1
        self.l2 = length2
        self.l3 = length3
        self.l4 = length4

        self.fig, self.ax = plt.subplots()
        self.ax.axis('equal')
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(-150, 100)
        self.robot_lines = []

        self.act_xPos = 10
        self.act_yPos = 20

        def mouse_move(event):
            #check if the mouse is in the self.ax and the mouse is pressed
            if event.xdata and event.ydata and event.button == 1:
                self.act_xPos, self.act_yPos = event.xdata, event.ydata
            #print("MousePos",self.act_xPos, self.act_yPos)
            

        plt.connect('motion_notify_event', mouse_move)

        self.lines = []

    def intersect_circles(self, A, a, B, b):
        AB0 = B[0] - A[0]
        AB1 = B[1] - A[1]

        c = math.sqrt(AB0**2 + AB1**2)

        if c == 0:
            return [0, 0], [0, 0]

        x = (a**2 + c**2 - b**2) / (2 * c)
        y = a**2 - x**2

        if y < 0:
            return [0, 0], [0, 0]

        if y > 0:
            y = math.sqrt(y)
        ex0 = AB0 / c
        ex1 = AB1 / c
        ey0 = -ex1
        ey1 = ex0
        Q1x = A[0] + x * ex0
        Q1y = A[1] + x * ex1

        if y == 0:
            return [Q1x, Q1y], [0, 0]

        Q2x = Q1x - y * ey0
        Q2y = Q1y - y * ey1

        Q1x += y * ey0
        Q1y += y * ey1

        return [Q1x, Q1y], [Q2x, Q2y]


    def calculate_arm_coordinates(self, angle_m1, angle_m2):
        coordinate_m1_2 = (
            math.cos(angle_m1 * (math.pi) / 180) * self.l1 + self.koordinate_m1[0],
            math.sin(angle_m1 * (math.pi) / 180) * self.l1 + self.koordinate_m1[1]
        )
        coordinate_m2_2 = (
            math.cos(angle_m2 * (math.pi) / 180) * self.l2 + self.koordinate_m2[0],
            math.sin(angle_m2 * (math.pi) / 180) * self.l2 + self.koordinate_m2[1]
        )
        S_coordinate_1, S_coordinate_2 = self.intersect_circles(
            coordinate_m1_2, self.l3, coordinate_m2_2, self.l4
        )
        return self.koordinate_m1, self.koordinate_m2, coordinate_m1_2, coordinate_m2_2, S_coordinate_1, S_coordinate_2

    def get_motor_angles(self, x, y):
        # Berechnen Sie die Winkel der Motoren
        self.c1 = (self.koordinate_m1[0] - x)**2 + (self.koordinate_m1[1] - y)**2
        self.c2 = (self.koordinate_m2[0] - x)**2 + (self.koordinate_m2[1] - y)**2

        if self.c1 > (self.l1 + self.l3)**2:
            self.c1 = (self.l1 + self.l3)**2
        if self.c1 < (self.l1 - self.l3)**2:
            self.c1 = (self.l1 - self.l3)**2

        if self.c2 > (self.l2 + self.l4)**2:
            self.c2 = (self.l2 + self.l4)**2
        if self.c2 < (self.l2 - self.l4)**2:
            self.c2 = (self.l2 - self.l4)**2

        #print(self.c1, self.c2)

        angle_m1 = -math.acos((self.l1**2+self.c1-self.l3**2)/(2*self.l1*math.sqrt(self.c1)))
        angle_m2 = math.acos((self.l2**2+self.c2-self.l4**2)/(2*self.l2*math.sqrt(self.c2)))
        #calculate the radian to degree
        angle_m1 = angle_m1 * 180 / math.pi
        angle_m2 = angle_m2 * 180 / math.pi

        c1=math.sqrt(self.c2)
        c2=math.sqrt(self.c2)

        #add the turn angle
        angle_m1 = angle_m1 - (math.acos(max(min(x-self.koordinate_m1[0], c1),-c1)/c1) * 180 / math.pi)
        angle_m2 = angle_m2 - (math.acos(max(min(x-self.koordinate_m2[0], c2),-c2)/c2) * 180 / math.pi)

        return angle_m1, angle_m2


    def show(self, coo_1, coo_2, coo_3, coo_4, coo_5, coo_6):
        # Löschen Sie alle alten Linien
        for line in self.lines:
            line.remove()
        self.lines = []

        # Zeichnen Sie die neuen Linien
        self.lines.extend(self.ax.plot([coo_1[0], coo_3[0]], [coo_1[1], coo_3[1]], marker='o', color='red'))
        self.lines.extend(self.ax.plot([coo_2[0], coo_4[0]], [coo_2[1], coo_4[1]], marker='o' , color='red'))
        self.lines.extend(self.ax.plot([coo_3[0], coo_5[0]], [coo_3[1], coo_5[1]], marker='o', color='green'))
        self.lines.extend(self.ax.plot([coo_3[0], coo_6[0]], [coo_3[1], coo_6[1]], marker='o', color='green'))
        self.lines.extend(self.ax.plot([coo_4[0], coo_5[0]], [coo_4[1], coo_5[1]], marker='o',  color='green'))
        self.lines.extend(self.ax.plot([coo_4[0], coo_6[0]], [coo_4[1], coo_6[1]], marker='o',  color='green'))

        #self.ax.plot(self.act_xPos, self.act_yPos, marker='o', color='blue')

    
    def set_servo_angles_reality(self, angle1, angle2):
        #set the servo position

        # Read joystick axes2
        axis_x2 = angle1
        axis_y2 = angle2

        #defining the angles from the zero angle to the start angle of the simulation
        ServoID3_angle = 45
        ServoID5_angle = -45 

        # Map joystick values to servo positions
        servo_pos1_2 = -axis_x2-ServoID3_angle
        servo_pos2_2 = -axis_y2-ServoID5_angle

        #make sure the angle is in the save range
        # if not servo_pos1_2 > 35:
        #     servo_pos1_2 = 35
        # if not servo_pos2_2 < 230:
        #     servo_pos2_2 = 230
        # elif not servo_pos2_2 > 90:
        #     servo_pos2_2 = 90

        # else:
        #Leg couple
        # ServoManager.setServoPosSpeedAcc(3, 360-servo_pos2_2, 4000)
        # ServoManager.setServoPosSpeedAcc(4, servo_pos2_2, 4000)#

        # ServoManager.setServoPosSpeedAcc(5, 360-servo_pos1_2, 4000)
        # ServoManager.setServoPosSpeedAcc(6, servo_pos1_2, 4000)#

        #syncronize the servos
        #ServoManager.setGroupSync_ServoPosSpeedAcc([3, 4, 5, 6], 
        #                                     [servo_pos1_2, servo_pos1_2, servo_pos2_2, servo_pos2_2])

        time.sleep(0.1)
    


def _map(value, in_min, in_max, out_min, out_max):
    """Maps a value from one range to another"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min




def update_mouse_koo(frame):
    global step
    #Read the coordinate of the cursor on the self.ax
    #get the angles of the motors
    # if step >= len(movement_coo_list):
    #     step = 0
    # angle1, angle2 = robot_arm.get_motor_angles(movement_coo_list[step][0], movement_coo_list[step][1])
    # step += 1

    angle1, angle2 = robot_arm.get_motor_angles(robot_arm.act_xPos, robot_arm.act_yPos)
    print(angle1, angle2)
    #angle1, angle2 = 0,0

    all_coorid = robot_arm.calculate_arm_coordinates(angle1, angle2)
    robot_arm.show(*all_coorid)

    #show it in the reality
    #set the servo position

    robot_arm.set_servo_angles_reality(angle1, angle2)
    time.sleep(0.1)

def update_movement_coo_list(frame):
    global step
    #Read the coordinate of the cursor on the self.ax
    #get the angles of the motors
    if step >= len(movement_coo_list):
        step = 0
    angle1, angle2 = robot_arm.get_motor_angles(movement_coo_list[step][0], movement_coo_list[step][1])
    step += 1

    #angle1, angle2 = robot_arm.get_motor_angles(robot_arm.act_xPos, robot_arm.act_yPos)
    print(angle1, angle2)
    #angle1, angle2 = 0,0

    all_coorid = robot_arm.calculate_arm_coordinates(angle1, angle2)
    robot_arm.show(*all_coorid)

    #show it in the reality
    #set the servo position

    robot_arm.set_servo_angles_reality(angle1, angle2)
    time.sleep(0.1)


if __name__ == "__main__":
    robot_arm = WholeRobot((0, 0), (27, 5), 70, 70, 70, 65)

    step = 0
    #movement_coo_list = [(85, -52),   (-45, 76)]#(26, -97),(10, -110),
    movement_coo_list = [(35, -40),  (-6, 30), (-40, -70), (-6, 32)]#(-6, 40), (45, -35),

    

    # Erstellen der Animation
    ani = FuncAnimation(robot_arm.fig, update_mouse_koo, frames=8, repeat=True, blit=False)
    
    plt.show()
