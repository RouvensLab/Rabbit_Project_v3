from scservo_sdk import *

import os
import time
import sys
import threading
import numpy as np
from prettytable import PrettyTable

from PySide6.QtWidgets import QApplication


class ServoControler:
    def __init__(self, Init_UI = False):
        # some global defined variables. Not changable-------------------
        self.SERIAL_PORT = 'COM3'
        self.BAUDRATE = 115200
        self.SCS1_ID = 1
        self.SCS2_ID = 5
        self.SCS_MOVING_ACC = 255
        self.SCS_MOVABLE_RANGE = (0, 4094)
        self.ADDR_STS_GOAL_POSITION = 42
        self.ADDR_STS_GOAL_ACC = 41
        self.ADDR_STS_GOAL_SPEED = 46
        self.protocol_end = 0
        self.initializing_pause = 3

                #-------EPROM(read & write)--------These values tell what should be changed. EveryServo information has a different address.
        self.SMS_STS_ID = 5
        self.SMS_STS_BAUD_RATE = 6
        self.SMS_STS_MIN_ANGLE_LIMIT_L = 9
        self.SMS_STS_MIN_ANGLE_LIMIT_H = 10
        self.SMS_STS_MAX_ANGLE_LIMIT_L = 11
        self.SMS_STS_MAX_ANGLE_LIMIT_H = 12
        self.SMS_STS_CW_DEAD = 26
        self.SMS_STS_CCW_DEAD = 27
        self.SMS_STS_OFS_L = 31
        self.SMS_STS_OFS_H = 32
        self.SMS_STS_MODE = 33

        self.SMS_STS_PROTECTIVE_TORQUE = 34
        self.SMS_STS_PROTECTIVE_TIME = 35
        self.SMS_STS_OVERLOAD_TORQUE = 36


        # Servo protocol details
        self.ADDR_SCS_TORQUE_ENABLE = 40
        self.ADDR_STS_GOAL_ACC = 41
        self.ADDR_STS_GOAL_POSITION = 42
        self.ADDR_STS_GOAL_SPEED = 46
        self.ADDR_SCS_PRESENT_POSITION = 56
        self.SCS_MINIMUM_POSITION_VALUE = 100
        self.SCS_MAXIMUM_POSITION_VALUE = 4000
        self.SCS_MOVING_STATUS_THRESHOLD = 20
        self.SCS_MOVING_ACC = 0
        self.protocol_end = 0            # SCServo bit end(STS/SMS=0, SCS=1)
        self.scs_goal_position = [self.SCS_MINIMUM_POSITION_VALUE, self.SCS_MAXIMUM_POSITION_VALUE]

        #Servo range properies
        self.servo_parameters = {
            "PresPos": 4094, "PresSpd": 4094, "Load": 1023, 
            "Voltage": 100, "Current": 0.01, "Temperature": 100, 
            "torque_enable": 1, "LastUpdate": None
        }



        # Servo properties
        self.servos_info_limitation = {
            1: {"min": 0, "max": 360, "orientation": 1},
            2: {"min": 0, "max": 360, "orientation": 0},
            3: {"min": 0, "max": 360, "orientation": 0},
            4: {"min": 0, "max": 360, "orientation": 1},
            5: {"min": 0, "max": 360, "orientation": 0},
            6: {"min": 0, "max": 360, "orientation": 1},
            7: {"min": 0, "max": 360, "orientation": 0},
            8: {"min": 0, "max": 360, "orientation": 1},
        }



        # Changable variables with important information-----------------
        self.start_time = time.time()

            #present information of the servos
        self.pres_servos_infos = {}
        for servo_id in self.servos_info_limitation.keys():
            self.pres_servos_infos[servo_id] = {"PresPos": 0, "PresSpd": 0, "Load": 0, "Voltage": 0, "Current": 0, "Temperature": 0, "torque_enable": 1, "LastUpdate": 0}

        self.pres_servos_actions = {}
        for servo_id in self.servos_info_limitation.keys():
            self.pres_servos_actions[servo_id] = {"angle": 0, "speed": 0, "acc": 0, "LastUpdate": 0}

        self.recorded_servo_infos = []

            #log information
        max_log_timespace = 60 #in seconds
        self.log_infos = {servo_id: np.array([[0,0,0,0,0,0,0,0]]) for servo_id in self.servos_info_limitation.keys()}
        self.log_infos["key_list"] = ["PresPos", "PresSpd", "Load", "Voltage", "Current", "Temperature", "torque_enable", "LastUpdate"]
        self.log_actions = {servo_id: np.array([[0,0,0,0]]) for servo_id in self.servos_info_limitation.keys()}
        self.log_actions["key_list"] = ["angle", "speed", "acc", "LastUpdate"]

        



        # Initialize PortHandler and PacketHandler instances
        self.portHandler = PortHandler(self.SERIAL_PORT)
        self.packetHandler = PacketHandler(self.protocol_end)
        # Initialize GroupSyncWrite instance
        self.groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_STS_GOAL_POSITION, 2)
        

        # connect to the port
        self.portHandler.openPort()
        self.portHandler.setBaudRate(self.BAUDRATE)
        time.sleep(self.initializing_pause)

        #if Init_UI:
        if Init_UI:
            #threading for the UI
            
            self._initUI_thread = threading.Thread(target=self._initUI, args=(self._give_servo_info, ))
            self._initUI_thread.start()
        else:
            self.app = None
            self.window = None

        #Observer and Safety Thread
        # self.servo_maintainer = threading.Thread(target=self._servo_maintainer)
        # self.servo_maintainer.start()


            
            
    #UI Panal
    def _initUI(self, recorded_servo_infos):
        # self.app = QApplication(sys.argv)
        # self.window = ServoControlApp(recorded_servo_infos)
        # self.window.show()
        sys.exit(self.app.exec())

    def _give_servo_info(self):
        '''
        Give the servo information to the UI
        '''
        return self.log_infos
    def _sent_servo_info(self):
        '''
        Sent the servo information to the UI
        '''
        self.window.update_total_plot(self.log_infos)

    #Observer and Safety Thread
    def _servo_maintainer(self):
        '''
        This function is used to actualize the servo information
        '''
        while True:
            self.actualize_servo_infos()
            #print("Servo Infos actualized")
            time.sleep(0.5)
    



    #general functions as tools (mapping)--------------------------------

    def _map(self, x, in_min, in_max, out_min, out_max):
        return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


    #reading Data -----------------------------------------------------
    def read_servo_data(self, servo_id, address, length):
        if length == 1:
            data, scs_comm_result, scs_error = self.packetHandler.read1ByteTxRx(self.portHandler, servo_id, address)
        elif length == 2:
            data, scs_comm_result, scs_error = self.packetHandler.read2ByteTxRx(self.portHandler, servo_id, address)
        elif length == 4:
            data, scs_comm_result, scs_error = self.packetHandler.read4ByteTxRx(self.portHandler, servo_id, address)
        else:
            raise ValueError("Invalid length specified")
        
        # Check if the returned values are valid
        # if scs_comm_result != 0 or scs_error != 0:
        #     raise Exception(f"Communication error: {scs_comm_result}, Error: {scs_error}")
        return data
    
    def get_Servo_info(self, SCS_ID):
        '''
        Get all the information of the servos
        '''

        Servo_info_list = {}

        # Read SC Servo present position and speed
        Servo_info_list["PresPos"] = self.read_servo_data(SCS_ID, SMS_STS_PRESENT_POSITION_L, 2)
        Servo_info_list["PresSpd"] = self.read_servo_data(SCS_ID, SMS_STS_PRESENT_SPEED_L, 2)
        

        # Read SC Servo present load
        #read the load
        Servo_info_list["Load"] =  self.read_servo_data(SCS_ID, SMS_STS_PRESENT_LOAD_L, 1)

        # Read SC Servo present voltage
        #read the voltage
        Servo_info_list["Voltage"] =  self.read_servo_data(SCS_ID, SMS_STS_PRESENT_VOLTAGE, 1)

        #read SC Servo present current
        Servo_info_list["Current"] =  self.read_servo_data(SCS_ID, SMS_STS_PRESENT_CURRENT_L, 1) * 0.0065 #convert to A from table excel one note
        # Read SC Servo present temperature
        #read the temperature
        Servo_info_list["Temperature"] =  self.read_servo_data(SCS_ID, SMS_STS_PRESENT_TEMPERATURE, 1)

        #read the torque enable
        Servo_info_list["torque_enable"] =  self.read_servo_data(SCS_ID, SMS_STS_TORQUE_ENABLE, 1)

        #last update date
        Servo_info_list["LastUpdate"] = time.time()-self.start_time
        #print(Servo_info_list)

        #actualize the global servos_infos dict with the new information
        self.pres_servos_infos[SCS_ID] = Servo_info_list

        def normalize_data(data, min_value, max_value):
            if max_value == None:
                return data
            return np.clip((data - min_value)/(max_value - min_value), 0, 1)

        #actualize the log_infos
        self.log_infos[SCS_ID] = np.concatenate((self.log_infos[SCS_ID], np.array([[normalize_data(Servo_info_list[key], 0, self.servo_parameters[key]) for key in self.log_infos["key_list"]]])), axis=0)

    def actualize_servo_infos(self):
        '''
        Read all the servo information parallel
        '''
        for servo_id in self.servos_info_limitation.keys():
            self.get_Servo_info(servo_id)
        return self.pres_servos_infos
    

    
    
    def syncRead(self, ID_List):
        '''
        Read the data from the servos
        '''
        
        # Initialize GroupSyncRead instance
        self.groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, SMS_STS_PRESENT_POSITION_L, 2)
        self.groupSyncRead.clearParam()
        for servo_id in ID_List:
            self.groupSyncRead.addParam(servo_id)
        scs_comm_result = self.groupSyncRead.txRxPacket()
        #print(f"OK: {scs_comm_result}")
        for servo_id in ID_List:
            self.groupSyncRead.isAvailable(servo_id, SMS_STS_PRESENT_POSITION_L, 2)
            data = self.groupSyncRead.getData(servo_id, SMS_STS_PRESENT_POSITION_L, 2)
            print(f"Servo ID: {servo_id} Data: {data}")
        self.groupSyncRead.clearParam()

        
    def syncRead_2(self, ID_List):
        """
        Read the data from the servos
        """
        # Define the addresses and lengths of the data to be read
        data_parameters = [
            {"name": "Speed", "address": SMS_STS_PRESENT_SPEED_L, "length": 2},
            {"name": "Position", "address": SMS_STS_PRESENT_POSITION_L, "length": 2},
            {"name": "Load", "address": SMS_STS_PRESENT_LOAD_L, "length": 1},
            {"name": "Voltage", "address": SMS_STS_PRESENT_VOLTAGE, "length": 1},
            {"name": "Current", "address": SMS_STS_PRESENT_CURRENT_L, "length": 2},
            {"name": "Temperature", "address": SMS_STS_PRESENT_TEMPERATURE, "length": 1},
            {"name": "TorqueEnable", "address": SMS_STS_TORQUE_ENABLE, "length": 1}
        ]

        # Initialize dictionary to store the read data for each servo
        servo_data = {servo_id: {} for servo_id in ID_List}

        # Read each type of data
        for param in data_parameters:
            groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, param["address"], param["length"])
            groupSyncRead.clearParam()

            # Add each servo ID to the sync read instance
            for servo_id in ID_List:
                groupSyncRead.addParam(servo_id)

         
            # Transmit packet and receive the result
            scs_comm_result = groupSyncRead.txRxPacket()
            
            if scs_comm_result != COMM_SUCCESS:
                print(scs_comm_result)
                print(f"Failed to read {param['name']} data for servos. Error: {self.packetHandler.getTxRxResult(scs_comm_result)}")
            
            time.sleep(1)
            
            # Fetch the data for each servo
            for servo_id in ID_List:
                if groupSyncRead.isAvailable(servo_id, param["address"], param["length"]):
                    data = groupSyncRead.getData(servo_id, param["address"], param["length"])
                    
                    # If length is 2, it means we have a low and high byte
                    if param["length"] == 2:
                        low_byte = data & 0xFF
                        high_byte = (data >> 8) & 0xFF
                        combined_data = (high_byte << 8) | low_byte
                        servo_data[servo_id][param["name"]] = combined_data
                    else:
                        servo_data[servo_id][param["name"]] = data

                    #print(f"Servo ID: {servo_id} {param['name']} Data: {servo_data[servo_id][param['name']]}")
                else:
                    print(f"Failed to get {param['name']} data for Servo ID: {servo_id}")


            
       
        return servo_data

    def print_servo_data_table(self,servo_data):
        """
        Print the servo data in a table format with parameters as column headers and servo IDs as rows
        """
        # Define the column headers
        headers = ["Servo ID", "Position", "Speed", "Load", "Voltage", "Current", "Temperature", "TorqueEnable"]
        
        # Create the table
        table = PrettyTable()
        table.field_names = headers

        # Add rows to the table
        for servo_id, data in servo_data.items():
            row = [
                servo_id,
                data.get("Position", "N/A"),
                data.get("Speed", "N/A"),
                data.get("Load", "N/A"),
                data.get("Voltage", "N/A"),
                data.get("Current", "N/A"),
                data.get("Temperature", "N/A"),
                data.get("TorqueEnable", "N/A")
            ]
            table.add_row(row)

        # Print the table
        print(table)

    def ScanServos(self):
        '''
        Scan for servos and returns a list with all the active servos and their id
    
        '''
        #self.servo_maintainer.stop()#stopt the thread, so that the servos can be initialized.
        if self.portHandler is None:
            print("Error: portHandler is not initialized.")
            return {}
        # Initialize the port with the baud rate
        print(f"Scanning for servos in the range 0 to 20 at {self.BAUDRATE} baud rate...")
        servo_pres = {}
        for servo_id in range(9):
            model_number ,result, error = self.packetHandler.ping(self.portHandler, servo_id)    

            if result == COMM_SUCCESS:
                print(f"Servo ID: {servo_id}: Model Number {model_number}")
                servo_pres[servo_id] = True
            else:
                print(f"Servo ID: {servo_id} Not present")
                servo_pres[servo_id] = None

        #self.servo_maintainer.continue_()#continue the thread

        return servo_pres
    




    #controling the the servo motors
    def syncCtrl(self,ID_List, Speed_List, Goal_List):
        positionList = []

        for i in range(0, len(ID_List)):
            #set goal speed for each servo
            try:
                scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, ID_List[i], self.ADDR_STS_GOAL_SPEED, Speed_List[i])  #Reg 46 Running Speed
            except:
                time.sleep(0.1)
                scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, ID_List[i], self.ADDR_STS_GOAL_SPEED, Speed_List[i])

            positionBuffer = [SCS_LOBYTE(Goal_List[i]), SCS_HIBYTE(Goal_List[i])]
            positionList.append(positionBuffer)
            
        
        for i in range(0, len(ID_List)):
            self.scs_addparam_result = self.groupSyncWrite.addParam(ID_List[i], positionList[i])
        
        scs_comm_result = self.groupSyncWrite.txPacket()
        #print(f"OK: {scs_comm_result}")
        self.groupSyncWrite.clearParam()

    def setSingleServoPosSpeedAcc(self, Servo_ID, angle, speed, safety_check = False):
        '''
        Set the position of a single servo
        '''
        if safety_check:
            #check if the angle is out of the limitation
            angle = self.safety_check(Servo_ID, angle)

        if self.servos_info_limitation[Servo_ID]["orientation"] == 1:
            servo_pos = int(self._map(angle, 180, -180, *self.SCS_MOVABLE_RANGE))
        else:
            servo_pos = int(self._map(angle, -180, 180, *self.SCS_MOVABLE_RANGE))
           

        #print(f"Servo ID: {Servo_ID} Angle: {angle}, Servo Pos: {servo_pos}")
        
        #actualize the global servos_actions dict with the new information
        self.pres_servos_actions[Servo_ID] = {"angle": angle, "speed": speed, "acc": 255}


        #write the angle and speed
        try:
            scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, Servo_ID, self.ADDR_STS_GOAL_SPEED, speed)  #Reg 46 Running Speed
        except:
            time.sleep(0.1)
            scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, Servo_ID, self.ADDR_STS_GOAL_SPEED, speed)

        positionBuffer = [SCS_LOBYTE(servo_pos), SCS_HIBYTE(servo_pos)]

        self.scs_addparam_result = self.groupSyncWrite.addParam(Servo_ID, positionBuffer)

        #make safety_limitation
        #self.setSafetyLimitation(Servo_ID)
    
    def run_sync_write_commands(self):
        '''
        Run the sync write commands
        '''
        scs_comm_result = self.groupSyncWrite.txPacket()
        #print(f"OK: {scs_comm_result}")
        self.groupSyncWrite.clearParam()


    def safety_check(self, Servo_ID, angle):
        '''
        Check if the servos are in the right position range
        '''
        if len(list(self.pres_servos_actions.keys())) == 8:
            
            #check if servo 3 and 5 don't collide
            max_angle_dif = -25
            min_angle_dif = -50

            if Servo_ID == 3:
                angle_dif = angle - self.pres_servos_actions[5]["angle"]
                if angle_dif > max_angle_dif:
                    return self.pres_servos_actions[5]["angle"] + max_angle_dif
                elif angle_dif < min_angle_dif:
                    return self.pres_servos_actions[5]["angle"] + min_angle_dif
                else:
                    return angle
            elif Servo_ID == 5:
                angle_dif = self.pres_servos_actions[3]["angle"] - angle
                if angle_dif > max_angle_dif:
                    return self.pres_servos_actions[3]["angle"] - max_angle_dif
                elif angle_dif < min_angle_dif:
                    return self.pres_servos_actions[3]["angle"] - min_angle_dif
                else:
                    return angle
                
            #check if servo 4 and 6 don't collide
            elif Servo_ID == 4:
                angle_dif = angle - self.pres_servos_actions[6]["angle"]
                #print(angle_dif)
                if angle_dif > max_angle_dif:
                    return self.pres_servos_actions[6]["angle"] + max_angle_dif
                elif angle_dif < min_angle_dif:
                    return self.pres_servos_actions[6]["angle"] + min_angle_dif
                else:
                    return angle
                
            elif Servo_ID == 6:
                angle_dif = self.pres_servos_actions[4]["angle"] - angle
                if angle_dif > max_angle_dif:
                    return self.pres_servos_actions[4]["angle"] - max_angle_dif
                elif angle_dif < min_angle_dif:
                    return self.pres_servos_actions[4]["angle"] - min_angle_dif
                else:
                    return angle
                
            else:
                return angle
        else:
            return angle
        
    def safety_mode(self, max_load = 1000):
        '''
        As soon as the servos have a load above a certain value, the servos are switched off
        '''
        #looks that the servos_infos are up to date
        
        for ServoID in self.pres_servos_infos.keys():
            if time.time()-self.start_time - self.pres_servos_infos[ServoID]["LastUpdate"] > 0.5:
                self.actualize_servo_infos()
            if self.pres_servos_infos[ServoID]["Load"] > max_load:
                self.torque_state(ServoID, False)
            else:
                self.torque_state(ServoID, True)
        
    def setGroupSync_ServoPosSpeedAcc(self, Servo_IDs:list, angles:list, speed = 4000, accs = 255, safety_mode = False):
        '''
        Set the position of the servos synchronical

        accs: is not used in this version
        '''
        for i in range(len(Servo_IDs)):
            self.pres_servos_actions[Servo_IDs[i]] = {"angle": angles[i], "speed": speed, "acc": accs}
        for i in range(len(Servo_IDs)):
            self.setSingleServoPosSpeedAcc(Servo_IDs[i], angles[i], speed)

        #run the syncwrite commands
        self.run_sync_write_commands()
    
        #safety force
        if safety_mode:
            self.safety_mode()

    def torque_state(self, Servo_ID, torque_state):
        """
        Writes the torque enable value to the specified servo.

        Args:
            scs_id (int): The ID of the servo.
            torque_enable (int): The torque enable value to be written.

        Returns:
            int: The result of the write operation.
        """
        scs_comm_result, scs_error = self.packetHandler.write1ByteTxRx(self.portHandler, Servo_ID, self.ADDR_SCS_TORQUE_ENABLE, torque_state)

    def WriteOFS(self, scs_id, ofs):
        """
        Writes the offset value to the specified servo motor.

        Parameters:
        - scs_id (int): The ID of the servo motor.
        - ofs (int): The offset value to be written.

        Returns:
        - int: The result of the write operation.

        """
        self.LockEprom(scs_id)
        self.packetHandler.write2ByteTxRx(self.portHandler, scs_id, SMS_STS_OFS_L, ofs)
        self.unLockEprom(scs_id)

    def LockEprom(self, scs_id):
        return self.packetHandler.write1ByteTxRx(self.portHandler, scs_id, SMS_STS_LOCK, 1)

    def unLockEprom(self, scs_id):
        return self.packetHandler.write1ByteTxRx(self.portHandler, scs_id, SMS_STS_LOCK, 0)

    def WriteMiddlePos(self, scs_id, angle, type_degree = True):
        '''
        Set a given old angle to the new middle position. So 0 degree is the new middle position.

        scs_id: the id of the servo
        angle: the old angle in degree
        '''
        if type_degree:
            angle = self._map(angle, 0, 360, *self.SCS_MOVABLE_RANGE)
        return self.WriteOFS(scs_id, angle)
    
    def defineInitialPosition(self, Servo_ID):
        '''
            When calling this function, the servo will disable the torque, so you can move the servo to the initial position.
            After that, you can call the returned function to set the new middle position.
        '''
        def findInitialPosition():

            #set the default Middle position = 0
            self.WriteMiddlePos(Servo_ID, 0, type_degree=False)
            #read the current position of the servo
            middlepos = self.read_servo_data(Servo_ID, SMS_STS_PRESENT_POSITION_L, 2)
            print(middlepos)
            #check servos direction
            if self.servos_info_limitation[Servo_ID]["orientation"] == 1:
                middlepos = 2096 - middlepos

            #set the new middle position
            self.WriteMiddlePos(Servo_ID, middlepos, type_degree=False)
            time.sleep(0.5)
            #enable the torque
            self.torque_state(Servo_ID, True)

        self.torque_state(Servo_ID, False)
        return findInitialPosition
    
    def defineAllMiddlePositions(self):
        '''
        Set the middle position for all servos
        '''
        for Servo_ID in self.servos_info_limitation.keys():
            self.WriteMiddlePos(Servo_ID, 0, type_degree=False)

    def write_servo_data(self, servo_id, address, length, value):
        if length == 1:
            scs_comm_result, scs_error = self.packetHandler.write1ByteTxRx(self.portHandler, servo_id, address, value)
        elif length == 2:
            scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, servo_id, address, value)
        elif length == 4:
            scs_comm_result, scs_error = self.packetHandler.write4ByteTxRx(self.portHandler, servo_id, address, value)
        else:
            raise ValueError("Invalid length specified")
        
        # Check if the returned values are valid
        # if scs_comm_result != 0 or scs_error != 0:
        #     raise Exception(f"Communication error: {scs_comm_result}, Error: {scs_error}")
        return scs_comm_result, scs_error

    def setProtectiveTorque(self, Servo_ID, torque):
        '''
        Set the protective torque: Dont work!!!!!!!!!!!!!!!!
        '''
        #self.packetHandler.write2ByteTxRx(self.portHandler, Servo_ID, 34, torque)
        print("Protective torque", self.read_servo_data(Servo_ID, self.SMS_STS_PROTECTIVE_TORQUE, 1))
        print("Protective time", self.read_servo_data(Servo_ID, self.SMS_STS_PROTECTIVE_TIME, 1))
        print("Overload torque", self.read_servo_data(Servo_ID, self.SMS_STS_OVERLOAD_TORQUE, 1))

        #write the torque
        self.write_servo_data(Servo_ID, self.SMS_STS_PROTECTIVE_TORQUE, 1, torque)




    

if __name__=="__main__":
    if os.name == 'nt':
        import msvcrt
        print('nt')
        def getch():
            return msvcrt.getch().decode()
        
    Controler = ServoControler(Init_UI=False)
    Controler.ScanServos()

    


    while True:

        #Controler.setProtectiveTorque(4, 10)
        #Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0,0,20,20,0,0,0,0], 4000, 255)
        Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [-50,-50,-50,-50,-100,-100,-25,-25], 4000, 255)
        # #Controler.print_servo_data_table(Controler.syncRead_2([1,2,3,4,5,6,7,8]))
        time.sleep(2)
        Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0,0,0,0,0,0,0,0], 4000, 255)
        #print(Controler.syncRead_2([1,2,3,4,5,6,7,8]))
        time.sleep(2)
        # Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [50,50,50,50,50,50,50,50], 4000, 255)

        #Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0, 0, 0, -66.75000998629827, -66.75000998629827, 0, 0, -25.000010714391994, -25.000010714391994], 4000, 255)
        
        
        
        if os.name == 'nt' and msvcrt.kbhit():
            key = getch()
            if key.upper() == 'Q':
                # servo 4,3
                print("Terminating loop and turn of torque control.")
                for i in range(1, 9):       
                    Controler.torque_state(i, False)
                break

    
    # time.sleep(2) 
    # Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0,0,90,90,0,0,0,0], 4000, 255)
    
    # time.sleep(2)
    # Controler.syncRead([1,2,3,4,5,6,7,8])


    # print(Controler.actualize_servo_infos())

    # Controler.torque_state(4, False)
    # print("Torque off")
    # time.sleep(1)
    # Controler.torque_state(4, True)

    #read the current position of the servo
    # print(Controler.read_servo_data(4, SMS_STS_OFS_L, 2))
    # middlepos = Controler.read_servo_data(4, SMS_STS_PRESENT_POSITION_L, 2)
    # print(middlepos)
    # #set the new middle position
    # Controler.WriteMiddlePos(4, 0, type_degree=False)

    # print(Controler.read_servo_data(4, SMS_STS_OFS_L, 2))
    # print(Controler.read_servo_data(4, SMS_STS_PRESENT_POSITION_L, 2))

    # #test sync write
    # Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0,0,0,0,0,0,0,0], 4000, 255)
    # time.sleep(2)
    # print(Controler.read_servo_data(4, SMS_STS_PRESENT_POSITION_L, 2))

    
    #define the initial position
    # finish_func = Controler.defineInitialPosition(4)
    # print("Move the servo to the initial position")
    # input("Press any key to continue")
    # finish_func()
    #Controler.defineAllMiddlePositions()



    # Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0,0,0,0,0,0,0,0], 4000, 255)
    # time.sleep(2)
    # Controler.setGroupSync_ServoPosSpeedAcc([1,2,3,4,5,6,7,8], [0,0,180,180,0,0,0,0], 4000, 255)



