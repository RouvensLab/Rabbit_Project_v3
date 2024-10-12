from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from collections import deque



class OptimizerCallback(BaseCallback):
    """
    A custom callback that changes the learning rate depending on the average reward of the last 100 episodes.
    It is optimized for the `SAC-Imitation` algorithm. 
    :param log_dir: (str) The log directory
    :param eval_freq: (int) Evaluate the model every eval_freq call of the callback.
    :param best_model_save_path: (str) Path to save the best model
    :param save_replay_buffer: (bool) Save the replay buffer when saving the best model
    :param name_prefix: (str) Prefix to the best model name
    :param learningrate_adjustment: (bool) Adjust the learning rate based on the reward

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug

    """

    def __init__(self, log_dir, eval_freq:int=10000, best_model_save_path:str=None, save_replay_buffer:bool=True, name_prefix="best_rl_model", learningrate_adjustment=True, verbose=0):
        super(OptimizerCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.best_model_save_path = best_model_save_path
        self.save_replay_buffer = save_replay_buffer
        self.name_prefix = name_prefix
        self.learningrate_adjustment = learningrate_adjustment

        

        # number of episodes
        self.episode_num = 0
        self._plot = None

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_relative_rewards = deque(maxlen=100)
        self.slope_percent_list = deque(maxlen=100)



    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)





    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        #getting all the initial values of the environment. Afterwards delete the the env.
        # max_steps
        self.max_steps = self.training_env.get_attr("maxSteps")[0]
        print(f"Max steps: {self.max_steps}")
        # max_reward
        self.max_reward = 1
        # min_reward
        self.min_reward = 0
        # current batch size
        #self.batch_size = self.model.batch_size

        self.best_mean_reward = self.min_reward

        
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.episode_num += 1
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        #check if the current step is the evaluation step
        if self.n_calls % self.eval_freq == 0:
            #retrieve the training reward from the last 100 episodes from the rollout/ep_rew_mean
            mean_reward = np.mean(self.episode_rewards)
            print(f"Mean Reward: {mean_reward}")


            #Check if new best mean reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_save_path+f"\\{self.name_prefix}_best_mean_reward")
                # Save the replay buffer
                if self.save_replay_buffer:
                    self.model.save_replay_buffer(self.best_model_save_path+f"\\{self.name_prefix}_replay_buffer_best_mean_reward.pkl")

            if self.leaningrate_adjustment:
                mean_slope_percent = np.mean(self.slope_percent_list)
                new_learning_rate = None
                #if the relative reward is above a certain threshold, adjust the learning rate
                if mean_slope_percent > 0.001:
                    new_learning_rate = self.model.learning_rate * 0.9
                    print("Learning rate halved")
                elif mean_slope_percent < -0.0005:
                    new_learning_rate = self.model.learning_rate * 1.5
                    print("Learning rate reduced to 10%")
                else:
                    new_learning_rate = self.model.learning_rate * 0.95
                #set the new learning rate
                # Update the optimizer's learning rate
                # Update the learning rate for the critic optimizer
                # Adjust learning rate based on rewards
                if new_learning_rate:
                    self.model.learning_rate = np.clip(new_learning_rate, 1e-7, 1e-2)
                    self.model._setup_lr_schedule()

                print(f"New Learning Rate: {self.model.learning_rate}")

            #logging the Mean slope percent
            #self.logger.record("optimizer/Mean Reward Slope", mean_slope_percent)
                    

        return True
    
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Access the episode reward buffer (contains rewards from recent episodes)
        if len(self.model.ep_info_buffer) > 0:
            # Extract rewards for the most recent episode
            rewards = [info['r'] for info in self.model.ep_info_buffer]
            if rewards:
                ep_rew_mean = np.mean(rewards)
                #print(f"Episode Reward Mean: {ep_rew_mean}")
                self.episode_rewards.append(ep_rew_mean)
            #extract the ep_length for the most recent episode
            ep_length = [info['l'] for info in self.model.ep_info_buffer]
            if ep_length:
                ep_length_mean = np.mean(ep_length)
                #print(f"Episode Length Mean: {ep_length_mean}")
                self.episode_lengths.append(ep_length_mean)
            if ep_length and rewards:
                relative_reward = np.mean(rewards)/np.mean(ep_length)
                self.episode_relative_rewards.append(relative_reward)
                self.logger.record("optimizer/Relative Reward", relative_reward)
                #calculate the slope of the reward curve over the last 100 episodes
                #check that there are at least 100 episodes
                mean_slope_percent = (self.episode_relative_rewards[-1] - self.episode_relative_rewards[0])/self.max_reward
                self.slope_percent_list.append(mean_slope_percent)
                self.logger.record("optimizer/Mean Reward Slope", mean_slope_percent)
                #print(f"Mean Slope Percent: {mean_slope_percent}")

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    
