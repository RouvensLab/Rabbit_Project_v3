

import matplotlib.pyplot as plt
import numpy as np

max_reward = 1


def calculate_reward(distance, val_base):
    reward_total = np.exp(val_base * distance**2)
    return reward_total

fig, ax = plt.subplots()
#set title
plt.title("Reward Function")

#legends
ax.set_xlabel("Distance")
ax.set_ylabel("Reward")

#set the grid
ax.grid(True)

# #add a vertival line at x = 0.1
# ax.axvline(x=0.1, color='r', linestyle='--')
# #horizontal line at y = 1
# ax.axhline(y=1, color='g', linestyle='--')
x_distance_list = np.arange(0, 3.5, 0.01)


for ex_value in [-0.1, -1, -2, -5, -10, -20, -40]:
    y_reward_list = [calculate_reward(distance, ex_value) for distance in x_distance_list]
    ax.plot(x_distance_list, y_reward_list, label=f"Reward base={str(ex_value)}")

# Add legend
ax.legend()

#show the plot
plt.show()