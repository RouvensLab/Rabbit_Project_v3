import sys
import time
import os
import json
import ast
import threading
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QMessageBox, QSlider, QFileDialog, QLabel, QCheckBox
)

from PySide6.QtCore import Qt
from envs.force_imitation import CustomEnv
from stable_baselines3 import SAC

class ExtendedTableWidget(QTableWidget):
    """Table with extendable rows."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(9) # Time + 8 action values
        self.setHorizontalHeaderLabels(['Time', 'Action1', 'Action2', 'Action3', 'Action4', 'Action5', 'Action6', 'Action7', 'Action8'])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def get_dict(self):
        #get the whole data from the table and transform it to the timetable dictionary
        timetable = {}
        for row in range(self.rowCount()):
            time = float(self.item(row, 0).text())
            actions = [float(self.item(row, col).text()) for col in range(1, self.columnCount())]
            timetable[time] = actions
        return timetable
    
    def load_action_timetable(self, timetable):
        self.setRowCount(len(timetable))

        for row, (time, actions) in enumerate(timetable.items()):
            self.setItem(row, 0, QTableWidgetItem(str(time)))
            for col, action_value in enumerate(actions):
                self.setItem(row, col + 1, QTableWidgetItem(str(action_value)))

    def add_new_row(self):
        # Insert a new row at the current selected position, or at the end if no selection
        selected_row = self.currentRow()
        if selected_row == -1:  # If no row selected
            selected_row = None
        self.insert_empty_row(selected_row)
    
    def remove_selected_row(self):
        # Remove the selected row
        selected_row = self.currentRow()
        if selected_row != -1:
            self.removeRow(selected_row)

    def insert_empty_row(self, position=None):
        """Inserts a new empty row at the given position (or at the end if no position is specified)."""
        if position is None:
            # If no position specified, append a new row at the end
            position = self.rowCount()
        
        self.insertRow(position)
        # Insert default empty values (e.g., 0.0 for float) in the new row
        self.setItem(position, 0, QTableWidgetItem(str(0.0)))  # Time
        for col in range(1, self.columnCount()):
            self.setItem(position, col, QTableWidgetItem(str(0.0)))  # Default action values

    def highlight_active_row(self, row):
        # Highlight the active row
        for col in range(self.columnCount()):
            self.item(row, col).setBackground(Qt.green)
        # Unhighlight the other rows
        for other_row in range(self.rowCount()):
            if other_row != row:
                for col in range(self.columnCount()):
                    self.item(other_row, col).setBackground(Qt.white)

class EnvParameterEditor(QWidget):
    def __init__(self, env_param_kwargs, restart_callback):
        super().__init__()
        self.initUI(env_param_kwargs, restart_callback)
    def initUI(self, env_param_kwargs, restart_callback):
        self.restart_callback = restart_callback
        
        # Create a layout for the editor
        layout = QVBoxLayout()
        # shows a table with the current environment parameters
        self.table_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.table_layout.addWidget(self.table)
        self.load_env_parameters(env_param_kwargs)
        layout.addLayout(self.table_layout)

        # Add button to save the parameterchanges and restart/reopen the environment
        button_layout = QHBoxLayout()
        open_param_button = QPushButton("Open Environment", self)
        open_param_button.clicked.connect(self.open_Env_param_from_file)
        button_layout.addWidget(open_param_button)
        self.save_button = QPushButton("Save and Restart", self)
        self.save_button.clicked.connect(self.save_and_restart)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def load_env_parameters(self, env_param_kwargs):
        # Load the environment parameters into the table
        self.table.setRowCount(len(env_param_kwargs))
        for row, (param_name, param_value) in enumerate(env_param_kwargs.items()):
            self.table.setItem(row, 0, QTableWidgetItem(param_name))
            if isinstance(param_value, (str)):
                self.table.setItem(row, 1, QTableWidgetItem(str(f"'{param_value}'")))
            else:
                self.table.setItem(row, 1, QTableWidgetItem(str(param_value)))
    def get_env_parameters(self):
        # Get the environment parameters from the table
        env_param_kwargs = {}
        for row in range(self.table.rowCount()):
            param_name = self.table.item(row, 0).text()
            param_value = self.table.item(row, 1).text()
            # Try to convert the parameter value to a float or int if possible or as a json string
            # print(param_value)
            env_param_kwargs[param_name] = ast.literal_eval(param_value)
        print(env_param_kwargs)
        return env_param_kwargs

    def open_Env_param_from_file(self):
        # Opens a QFileDialog to select new environment parameters
        file_dialog = QFileDialog(self, "Open Environment Parameters", r"Models", "Model Files (info.txt)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            # Load the environment parameters from the selected txt file
            with open(file_path, "r") as f:
                parameter_text = f.read()
            #find the line that begins with "Env Parameters: " afterwards is a dictionary with the environment parameters
            env_param_str = parameter_text.split("Env Parameters: ")[1]
            env_param_str = env_param_str.split("\n")[0]
            print(env_param_str)
            #convert the string to a dictionary
            try:
                env_param_kwargs = ast.literal_eval(env_param_str)
            except:
                QMessageBox.warning(self, "Error", "Could not load the environment parameters from the file!")
                return

            self.load_env_parameters(env_param_kwargs)
            QMessageBox.information(self, "Parameters Loaded", "Environment parameters loaded successfully!")
        else:
            QMessageBox.warning(self, "Parameters Not Loaded", "No parameters loaded!")

    def save_and_restart(self):
        self.restart_callback(self.get_env_parameters())
    


class ActionTimetableEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.env_param_kwargs = {
        "ModelType": "SAC",
        "rewards_type": ["stability"],
        "observation_type": ["joint_forces", "rhythm"],
        "simulation_stepSize": 5,
        "obs_time_space": 2,
        "maxSteps": 360*5,
        "restriction_2D": False,
        "terrain_type": "flat",
        "recorded_movement_file_path_list": [r"expert_trajectories\fast_linear_springing_v1.json"]
    }
        self.env_startup_parms = {"render_mode": "human", "real_robot": False, "gui": True}
        self.env = CustomEnv(**self.env_startup_parms, **self.env_param_kwargs)
        self.env.ROS_Env.hung = True

        self.manual_exp = ManualExpert(sim_freq=5)
        self.loaded_model = None

        self.env_param_editor = None

        self.env_pause = False
        self.simulation_thread = None
        self.initUI()
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.start()
        #self.simulation_thread.join()

    def initUI(self):
        layout = QVBoxLayout()
        layout1 = QHBoxLayout()


        #Bar that shows the active model. Also you can select a model file.
        bar_layout = QHBoxLayout()
        #load environment parameter button
        self.env_param_button = QPushButton("Open Environment Parameter Editor", self)
        self.env_param_button.clicked.connect(self.open_Env_param)
        bar_layout.addWidget(self.env_param_button)

        # Load model button
        self.model_loadButton = QPushButton("No Model Loaded", self)
        self.model_loadButton.clicked.connect(self.open_model_file)
        bar_layout.addWidget(self.model_loadButton)

        # Name of loaded model
        self.model_name_label = QLabel(self)
        self.model_name_label.setText("Model Name: Manual Expert")
        bar_layout.addWidget(self.model_name_label)

        # Activation checkbox for model
        self.button_mod_active = QCheckBox("Model Active", self)
        bar_layout.addWidget(self.button_mod_active)


        # Real robot Box
        real_robot_layout = QVBoxLayout()
        # Real robot checkbox
        self.button_real_robot = QCheckBox("Real Robot", self)
        self.button_real_robot.clicked.connect(self.toggle_real_robot)
        real_robot_layout.addWidget(self.button_real_robot)
        # checkbox for savety mode or direct mode
        self.button_real_robot_savety = QCheckBox("Savety Mode", self)
        real_robot_layout.addWidget(self.button_real_robot_savety)
        
        bar_layout.addLayout(real_robot_layout)
        
        layout.addLayout(bar_layout)



        # Table for action timetable
        self.table_layout = QVBoxLayout()
        self.table = ExtendedTableWidget(self)
        self.table.load_action_timetable(self.manual_exp.action_timetable)
        self.table.cellChanged.connect(self.on_table_cell_changed)
        self.table_layout.addWidget(self.table)



        layout1.addLayout(self.table_layout, stretch=3)

        # Add Buttons
        button_layout = QVBoxLayout()


        #add new row button
        self.add_row_button = QPushButton("Add New Row", self)
        self.add_row_button.clicked.connect(self.table.add_new_row)
        button_layout.addWidget(self.add_row_button)
        #remove row button
        self.remove_row_button = QPushButton("Remove Selected Row", self)
        self.remove_row_button.clicked.connect(self.table.remove_selected_row)
        button_layout.addWidget(self.remove_row_button)

        self.copy_button = QPushButton("Copy Timetable to Clipboard", self)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(self.copy_button)

        #hanging or not hanging button
        self.hung_button = QPushButton("hung", self)
        self.hung_button.clicked.connect(self.hung_change)
        button_layout.addWidget(self.hung_button)

        # Pause/Unpause button
        self.pause_button = QPushButton("Pause/Unpause Simulation", self)
        self.pause_button.clicked.connect(self.pause_unpause_simulation)
        button_layout.addWidget(self.pause_button)

        # Restart button
        self.restart_button = QPushButton("Restart Simulation", self)
        self.restart_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.restart_button)

        #add a speed slider
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(0)
        button_layout.addWidget(self.speed_slider)

        layout1.addLayout(button_layout, stretch=1)

        layout.addLayout(layout1, stretch=1)
        self.setLayout(layout)

    def hung_change(self):
        self.env.ROS_Env.hung = not self.env.ROS_Env.hung

    def savety_mode_change(self, checked):
        self.env.robot_precise_mode = checked


    def on_table_cell_changed(self):
        # Update the action timetable when a cell is changed
        #update the timetable
        self.manual_exp.action_timetable = self.table.get_dict()

    def reset_simulation(self):        
        self.env.reset()

        #QMessageBox.information(self, "Restart", "Simulation restarted!")

    def pause_unpause_simulation(self):
        self.env_pause = not self.env_pause



    def open_Env(self, env_param_kwargs):
        self.env_param_kwargs = env_param_kwargs
        #close the current environment and the thread
        self.end_thread = True
        self.simulation_thread.join()
        if self.env:
            self.env.close()
            self.env = None

        #open a new environment
        self.env = CustomEnv(**self.env_startup_parms, **self.env_param_kwargs)
        self.env.ROS_Env.hung = False
        self.env.robot_precise_mode = self.button_real_robot_savety.isChecked()
        self.env_pause = True

        #if everything is ok, close the editor window
        if self.env_param_editor:
            self.env_param_editor.close()
            self.env_param_editor = None
            print("env_param_editor closed!")


        #open a new thread
        self.simulation_thread = None
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.start()
        print("Simulation started!")


        

    
    def open_Env_param(self):
        #open the environment parameter editor
        self.env_param_editor = EnvParameterEditor(self.env_param_kwargs, self.open_Env)
        self.env_param_editor.setWindowTitle("Environment Parameter Editor")
        self.env_param_editor.resize(600, 400)
        self.env_param_editor.show()

    def toggle_real_robot(self, checked):
        # Restart the simulation by wether leading or without loading the robot_env
        self.env_startup_parms["real_robot"] = checked
        self.open_Env(self.env_param_kwargs)


    def open_model_file(self):
        # Open a file dialog to select a model file
        file_dialog = QFileDialog(self, "Open Model File", r"Models", "Model Files (*.zip)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            # Load the model from the selected file
            self.loaded_model = SAC.load(file_path)
            self.button_mod_active.setChecked(False)
            #change color of the model button
            self.model_loadButton.setStyleSheet("background-color: green")
            self.model_name_label.setText(f"Model Name: {file_path.split('/')[-1]}")
            self.env_pause = True
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully!")
        else:
            self.loaded_model = None
            self.button_mod_active.setChecked(False)
            self.model_loadButton.setStyleSheet("background-color: red")
            self.model_name_label.setText(f"Model Name: Manual Expert")
            self.env_pause = True
            QMessageBox.warning(self, "Model Not Loaded", "No model loaded!")

    def run_simulation(self):
        # This thread runs as long as there are no errors.
        # As long as the environments fit the model, everything is fine.
        self.no_error = True
        self.end_thread = False

        while not self.end_thread and self.no_error:
            obs, info = self.env.reset()
            done = False

            while not done and not self.end_thread and self.no_error:
                if not self.env_pause:
                    if self.loaded_model is not None and self.button_mod_active.isChecked():
                        try:
                            action, pred_info = self.loaded_model.predict(obs)
                        except Exception as e:
                            self.no_error = False
                            print(f"Error in model prediction: {e}")
                    else:
                        try:
                            action, state = self.manual_exp.think_and_respond(obs, None, done)
                        except Exception as e:
                            self.no_error = False
                            print(f"Error in manual response: {e}")

                    # Highlight the active row
                    action_key_index = list(self.manual_exp.action_timetable.keys()).index(self.manual_exp.action_key)
                    self.table.highlight_active_row(action_key_index)

                    try:
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                    except Exception as e:
                        self.no_error = False
                        print(f"Error in environment step: {e}")
                    

                    time.sleep(self.speed_slider.value() / 10)
                else:
                    time.sleep(0.05)

        if not self.no_error:
            # Dialog if there is an error
            print("Error", "An error occurred! Simulation stopped! Probably the Environment does not fit the model!")
            # If there is an error, close the environment
            self.env.close()
            self.env = None
            
    def copy_to_clipboard(self):
        timetable_str = str(self.manual_exp.action_timetable)
        clipboard = QApplication.clipboard()
        clipboard.setText(timetable_str)
        QMessageBox.information(self, "Clipboard", "Action timetable copied to clipboard!")


class ManualExpert:
    def __init__(self, sim_freq= 5):
        self.sim_freq = sim_freq

        self.max_rhytm_size = 1000

        self.current_action = [0 for i in range(9)]
        self.action_key = 0

        #define an action handling list
        # sprinting v1
        # self.action_timetable = {
        #     0:  [-0, 0.0,   -0.5, 0.3,  -0.5, 0.3,    0.2, 0.2],
        #     0.2:  [1, 0.2,   -0.2, 0.3,   -0.2, 0.3,    -1, -1],
        #     0.5:  [1, 0.2,   -0.2, -0.4,   -0.2, -0.4,    -1, -1],
        #     0.6:  [-1, 0.2,   -0.4, 0.7,   -0.4, 0.7,    0.2, 0.2],
        #     0.7 :  [-1, 0.0,   -0.5, 0.7,   -0.5, 0.7,    0.2, 0.2],
            
        #     0.7: [-1, 0,   -0.4, 0,   -0.4, 0,    0, 0]
        # }

        # sprinting v2
        # self.action_timetable = {
        #     0:  [0.2, 0,   -0.4, 0.4,  -0.4, 0.4,    0.3, 0.3],
        #     0.2:  [1, 0,   -0., 0.4,   -0., 0.4,    -0.7, -0.7],
        #     0.4:  [1, 0,   -0., -0.6,   -0., -0.6,    -0.7, -0.7],
        #     0.5:  [-0.8, 0,   -0.1, -0.6,   -0.1, -0.6,    -0.7, -0.7],
        #     0.6:  [-0.8, 0,   -0.4, 0.7,   -0.4, 0.7,    0.4, 0.4],
        #     0.7 :  [0.2, 0,   -0.4, 0.4,  -0.4, 0.4,    0.3, 0.3],
            
        #     0.7: [-1, 0,   -0.4, 0,   -0.4, 0,    0, 0]
        # }

        #pusch sprint v1
        self.action_timetable = {0.0: [-0.5, 0.0, -0.2, 0.4, -0.2, 0.4, -0.5, -0.5], 0.2: [-0.5, 0.0, 0.5, -0.4, 0.5, -0.4, 1.0, 1.0], 0.45: [1.0, 0.0, 0.5, 0.6, 0.5, 0.6, 0.0, 0.0], 0.6: [1.0, 0.0, -0.2, 0.6, -0.2, 0.6, -0.5, -0.5], 0.7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

        #sitting
        # self.action_timetable = {
        #     0:  [-0.3, 0,   -0.5, 0.3,  -0.5, 0.3,    0, 0]
        # }



    def think_and_respond(self, obs_, state, done):
        timestep = obs_[-1]

        # Define the action based on the time table
        action_keytimes = list(self.action_timetable.keys())
        #get the action which is at the current time step
        current_time_sec = timestep*self.max_rhytm_size*0.01
        last_time = action_keytimes[-1]
        
        #get the action key which is over the current time
        self.action_key = [key for key in action_keytimes if key <= current_time_sec][-1]
        if self.action_key == last_time:
            next_time_step_size = -timestep
            self.action_key = 0
        else:
            next_time_step_size = self.sim_freq/self.max_rhytm_size
        
        #print("action_key1: ", self.action_key)
        #print("timestep", timestep, "current_time_sec: ", current_time_sec, "action_key: ", self.action_key, "next_time_step_size: ", next_time_step_size)

        action = self.action_timetable[self.action_key]+[next_time_step_size]
        state = obs_[:-1]
        return np.array(action), state
    
    def think_response_with_ndarray(self, obs_ndarray, state_ndarray, done_ndarray):
        actions = []
        states = []
        for obs_, done in zip(obs_ndarray, done_ndarray):
            action, state = self.think_and_respond(obs_, state_ndarray, done)
            actions.append(action)
            states.append(state)
        return np.array(actions), state_ndarray



if __name__=="__main__":
    #env = CustomEnv(ModelType="SAC", gui=True, render_mode="human", maxSteps=360*5, terrain_type="flat", rewards_type=["stability"], observation_type=["joint_forces", "joint_angles", "rhythm"], simulation_stepSize=5, restriction_2D=False, real_robot=False)

    
    # #shows an automatic controller
    # expert = ManualExpert(sim_freq=5)
    # for episode in range(10):
        
    #     obs, info = env.reset()
    #     done = False
    #     while not done:
    #         action, state = expert.think_and_respond(obs, None, done)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         #env.render()
    #         done = terminated or truncated
    # env.close()

    app = QApplication(sys.argv)

    gui = ActionTimetableEditor()
    gui.setWindowTitle("Action Timetable Editor")
    gui.resize(600, 400)
    gui.show()

    sys.exit(app.exec())

