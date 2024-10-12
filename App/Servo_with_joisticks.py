import time
import pygame
from envs.Servo_Controler_v4 import *  # Import your ServoControler class

# Initialize pygame and the joystick module
pygame.init()
pygame.joystick.init()

# Open the first joystick device
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Create an instance of the ServoControler class
ServoManager = ServoControler()

# Function to map joystick values to servo positions
def map_joystick_to_servo(value, min_value, max_value):
    # Map joystick value (ranging from -1 to 1) to servo position (between min and max values)
    return int((value + 1) / 2 * (max_value - min_value) + min_value)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Read joystick axes1
    axis_x1 = joystick.get_axis(0)  # Replace with the appropriate axis index for your joystick
    axis_y1 = joystick.get_axis(1)  # Replace with the appropriate axis index for your joystick

    # Read joystick axes2
    axis_x2 = joystick.get_axis(2)  # Replace with the appropriate axis index for your joystick
    axis_y2 = joystick.get_axis(3)  # Replace with the appropriate axis index for your joystick


    # Map joystick values to servo positions
    servo_pos1_1 = map_joystick_to_servo(axis_x1, 0, 180)
    servo_pos2_1 = map_joystick_to_servo(axis_y1, 0, 180)

    # Map joystick values to servo positions
    servo_pos1_2 = map_joystick_to_servo(axis_x2, 0, 180)
    servo_pos2_2 = map_joystick_to_servo(axis_y2, 0, 180)




    # Set servo positions

    if abs(axis_x1) < 0.01 and abs(axis_y1) < 0.01:
        #print("relaxing1")
        ServoManager.setSingleServoPosSpeedAcc(1, servo_pos1_1, 0,0)
        ServoManager.setSingleServoPosSpeedAcc(2, servo_pos1_1, 0,0)
    else:
        ServoManager.setSingleServoPosSpeedAcc(1, servo_pos1_1, 4000)
        ServoManager.setSingleServoPosSpeedAcc(2, servo_pos2_1, 4000)

    if abs(axis_x2) < 0.01 and abs(axis_y2) < 0.01:
        #Leg couple
        #print("relaxing2")
        ServoManager.setSingleServoPosSpeedAcc(3, servo_pos1_2, 0, 0)
        ServoManager.setSingleServoPosSpeedAcc(4, 180-servo_pos1_2, 0,0)

        ServoManager.setSingleServoPosSpeedAcc(5, servo_pos2_2, 0,0)
        ServoManager.setSingleServoPosSpeedAcc(6, 180-servo_pos2_2, 0,0)

    else:
        #Leg couple
        ServoManager.setSingleServoPosSpeedAcc(3, servo_pos1_2, 4000)
        ServoManager.setSingleServoPosSpeedAcc(4, 180-servo_pos1_2, 4000)

        ServoManager.setSingleServoPosSpeedAcc(5, servo_pos2_2, 4000)
        ServoManager.setSingleServoPosSpeedAcc(6, 180-servo_pos2_2, 4000)

    ServoManager.run_sync_write_commands()


    # Delay for smoother servo movement
    time.sleep(0.1)
