
import numpy as np
import PySide6.QtWidgets as QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QSlider, QWidget
import threading

class SliderWindow(QWidget):
    def __init__(self, sliders_count, slider_val_func):
        super().__init__()
        self.slider_val_func = slider_val_func
        layout = QVBoxLayout()
        self.sliders = []
        self.labels = []
        for _ in range(sliders_count):
            child_layout = QHBoxLayout()
            slider = QSlider()
            slider.setOrientation(Qt.Horizontal)
            slider.setMinimum(-10)
            slider.setMaximum(10)
            slider.setValue(0)
            slider.valueChanged.connect(self.on_slider_change)
            child_layout.addWidget(slider)
            self.sliders.append(slider)
            #label
            label = QtWidgets.QLabel()
            label.setText("0")
            child_layout.addWidget(label)
            self.labels.append(label)

            layout.addLayout(child_layout)

        self.setLayout(layout)
    
    def on_slider_change(self):
        new_slider_values = [slider.value()/10 for slider in self.sliders]
        self.slider_val_func(new_slider_values)
        #update the label
        for i, slider in enumerate(self.sliders):
            self.labels[i].setText(str(slider.value()/10))

class ManualControlEnv:
    def __init__(self, env):
        self.env = env
        #self.action_space = env.action_space
        #self.observation_space = env.observation_space
        self.timesteps = 0
        self.current_action = [0 for _ in range(8)]

    def reset(self):
        self.timesteps = 0

    def controller(self, obs):
        """This shows a window with 10 sliders that can be used to control the robot."""
        def get_slider_values(slider_values):
            self.current_action = slider_values

        app = QtWidgets.QApplication([])
        self.window = SliderWindow(8, get_slider_values)
        self.window.show()
        app.exec()
    
    def get_action(self):
        return self.current_action

def env_thread_func(env, expert):
    obs, info = env.reset()
    expert.reset()
    done = False
    while not done:
        action = expert.get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        env.ROS_Env.render()
        #done = terminated or truncated
    env.ROS_Env.close()

if __name__=="__main__":
    from envs.force_imitation import CustomEnv
    env = CustomEnv(ModelType="SAC", render_mode="fast", gui=True, rewards_type=["stability"], observation_type=["euler_array", "joint_angles", "joint_forces", "goal_orientation"], simulation_stepSize=5, real_robot=True)
    expert = ManualControlEnv(env)
    
    # Start the environment loop in a separate thread
    env_thread = threading.Thread(target=env_thread_func, args=(env, expert))
    env_thread.start()

    # Start the GUI in the main thread
    expert.controller(None)
    
    # Wait for the environment thread to finish
    env_thread.join()
