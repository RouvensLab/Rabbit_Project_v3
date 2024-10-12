import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator
from typing import Callable
from typing import Union, List

# Function to format x-axis in Millions
def millions(x, pos):
    return f'{int(x*1e-6)}M'

# Function to load the TensorBoard logs
def load_tensorboard_logs(logdir):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={
        event_accumulator.SCALARS: 0,  # Load all scalar data
    })
    ea.Reload()
    return ea

# Function to plot metrics from the TensorBoard logs
def plot_metrics(logdir:str, output_dir=None, show=False, everything_in_one=False):


    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    ea = load_tensorboard_logs(logdir)
    available_metrics = ea.Tags()['scalars']
    
    # # Set consistent style and font
    # plt.style.use('seaborn-whitegrid')

    if everything_in_one:

        fig_main, ax_main = plt.subplots(figsize=(10, 6))
        ax_main.xaxis.set_major_formatter(FuncFormatter(millions))
        ax_main.set_xlabel("Steps (in Millions)")
        ax_main.set_ylabel("relative Value")
        ax_main.set_title("Training Metrics", pad=15)
        #Grid and ticks
        ax_main.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax_main.minorticks_on()
        ax_main.tick_params(axis='both', which='both', direction='in')
        plt.tight_layout()

    
    for metric in available_metrics:
        events = ea.Scalars(metric)
        steps = [event.step for event in events]
        values = [event.value for event in events]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, values, label=metric, color='C0', linewidth=2)

        # Format x-axis to show steps in millions
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        ax.set_xlabel("Steps (in Millions)")
        ax.set_ylabel(metric)
        ax.set_title(f"Training: {metric}", pad=15)

        # Grid and ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in')

        # Adjust legend outside the plot
        ax.legend()

        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save the plot or show it
        if output_dir:
            #change name of the metric so there are no / in the name
            metric_name = metric.replace("/", "_")
            output_path = os.path.join(output_dir, f"{metric_name}.png")
            fig.savefig(output_path)

        if everything_in_one:
            #make relative values
            max_value = max(values)
            values = [value/max_value for value in values]
            ax_main.plot(steps, values, label=metric, linewidth=2)
            ax_main.legend()
            plt.tight_layout()
        
        if show:
            fig.show()

    if everything_in_one:
        if output_dir:
            output_path = os.path.join(output_dir, "all_metrics.png")
            fig_main.savefig(output_path)
        if show:
            plt.show()

def plot_beside(opposite_metrics:list, logdir, output_dir=None, show=False):
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Format x-axis to show steps in millions
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xlabel("Steps (in Millions)")
    #merge all the metrics name into one name.
    title_metric = " and ".join(opposite_metrics).replace("/", "_")
    ax.set_ylabel("y")
    ax.set_title(f"Training: {title_metric}", pad=15)

    # Grid and ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')


    ea = load_tensorboard_logs(logdir)
    available_metrics = ea.Tags()['scalars']

    for metric in available_metrics:
        if metric in opposite_metrics:
            events = ea.Scalars(metric)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            ax.plot(steps, values, label=metric, linewidth=2)


    # Adjust legend outside the plot
    ax.legend()
    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Save the plot or show it
    if output_dir:
        output_path = os.path.join(output_dir, f"{title_metric}.png")
        fig.savefig(output_path)

    if show:
        plt.show(block=True)

        

def cal_relative_reward(ea:event_accumulator.EventAccumulator) -> Union[List[int], List[float], str]:
    """
    This devides the mean reward by the mean episode length to get the relative reward.
    This is because of the fact that the reward is dependent on the episode length. Every step a reward is added to the episode reward.
    So we get the average reward that outputs the environment.
    """
    #ea = load_tensorboard_logs(logdir)
    #available_metrics = ea.Tags()['scalars']


    events = ea.Scalars("rollout/ep_rew_mean")
    steps = [event.step for event in events]
    values = [event.value for event in events]
    #get the values of the opposite metrics
    events = ea.Scalars("rollout/ep_len_mean")
    steps_opp = [event.step for event in events]
    values_opp = [event.value for event in events]
    #check if the steps are the same
    assert steps == steps_opp, "The steps of the metrics are not the same"
    #calculate the relative difference. For the calculation of the relative reward, without being influenced by the ep_len.
    relative_values = [value / values_opp[i] for i, value in enumerate(values)]

    return steps, relative_values, "relative_reward"

    

def general_plot(logdir:str, tranforming_func:Callable, output_dir=None, show=False):
    
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Adjust legend outside the plot
    ax.legend()
    # Tight layout to ensure everything fits
    plt.tight_layout()

    ea = load_tensorboard_logs(logdir)
    steps, values, other_metric = tranforming_func(ea)
    ax.plot(steps, values, label=other_metric, linewidth=2)

    # Format x-axis to show steps in millions
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_xlabel("Steps (in Millions)")
    ax.set_ylabel(other_metric)
    ax.set_title(f"Training: {other_metric}", pad=15)

    # Grid and ticks
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')



    # Save the plot or show it
    if output_dir:
        output_path = os.path.join(output_dir, f"{other_metric}.png")
        fig.savefig(output_path)

    if show:
        plt.show()

def plot_compare_logdirs(logdirs:List[str], name_list:List[str], transforming_func:Callable=None, output_dir=None, show=False, Names_in_title=False):
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'Arial',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    all_ea_logs = []
    same_metrics = []
    for logdir in logdirs:
        ea = load_tensorboard_logs(logdir)
        all_ea_logs.append(ea)
        available_metrics = ea.Tags()['scalars']
        if not same_metrics:
            same_metrics = available_metrics
        else:
            same_metrics = list(set(same_metrics).intersection(available_metrics))
        

    for metric in same_metrics:

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if Names_in_title:
            title = metric.replace("/", "_")+"_"+"_".join(name_list)
        else:
            title = metric.replace("/", "_")


        # Grid and ticks
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in')

        for id, ea in enumerate(all_ea_logs):
            if transforming_func:
                steps, values, title = transforming_func(ea)
                ax.plot(steps, values, label=name_list[id], linewidth=2)
            else:
                events = ea.Scalars(metric)
                steps = [event.step for event in events]
                values = [event.value for event in events]
                ax.plot(steps, values, label=name_list[id], linewidth=2)

        # Format x-axis to show steps in millions
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
        ax.set_xlabel("Steps (in Millions)")
        ax.set_ylabel(metric)
        ax.set_title(f"Training: {title}", pad=15)
        # Adjust legend outside the plot
        ax.legend()
        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save the plot or show it
        if output_dir:
            output_path = os.path.join(output_dir, f"{title}.png")
            fig.savefig(output_path)

        if show:
            plt.show()

        if transforming_func:
            break


    



# # Example usage
# logdir = r"Models\SACReinforceImitation_V2_v6\models\logs\SAC_0"  # Replace with your logs directory
#find the main dir of the Observed logdirs Model
# output_dir = os.path.join(*logdir.split("\\")[:-3], "graphs")
# print(output_dir)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# plot_metrics(logdir, output_dir, show=True, everything_in_one=True)
# plot_beside(["rollout/ep_rew_mean","rollout/ep_len_mean"], logdir, output_dir, show=True)
# general_plot(logdir, output_dir, show=True, tranforming_func=cal_relative_reward)

#compare 2 different logdirs
logdir_1 = r"Models\SACReinforceImitation_V2_v8\models\logs\SAC_0"  # Replace with your logs directory
logdir_2 = r"Models\SACReinforceImitation_V2_v6\models\logs\SAC_0"  # Replace with your logs directory
logdir_3 = r"Models\SACReinforceImitation_V2_v7\models\logs\SAC_0"  # Replace with your logs directory
output_dir = r"App\graphs\data"  # Optional: specify a directory to save plots
plot_compare_logdirs([logdir_1, logdir_2, logdir_3], ["Umgebung_1", "Umgebung_2", "Umgebung_3"], output_dir=output_dir, show=True, transforming_func=cal_relative_reward)

