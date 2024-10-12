from stable_baselines3 import SAC

trained_model_dir = r"Models\SACReinforceImitation_V2_v6\models\rl_model_18000000_steps.zip"
# Initialize the model
model = SAC.load(trained_model_dir)

# Move the policy network to the device
from torchsummary import summary

# Zeige die Architektur des Actor-Netzwerks
summary(model.policy.actor, input_size=(model.observation_space.shape[0],))

# Zeige die Architektur des Critic-Netzwerks
summary(model.policy.critic, input_size=[(model.observation_space.shape[0],), (model.action_space.shape[0],)])

# Zeige die Architektur des Value-Netzwerks
#summary(model.policy.value, input_size=(model.observation_space.shape[0],))

# Zeige die Architektur des Q-Netzwerks
#summary(model.policy.q_net, input_size=[(model.observation_space.shape[0],), (model.action_space.shape[0],)])

# import matplotlib.pyplot as plt
# import networkx as nx

# # Erstellt einen gerichteten Graphen
# G = nx.DiGraph()

# # Definiert die Schichten für das Actor-Netzwerk
# actor_layers = [
#     ("Input (Flatten)", "Linear (290 -> 512)"),
#     ("Linear (290 -> 512)", "ReLU"),
#     ("ReLU", "Linear (512 -> 256)"),
#     ("Linear (512 -> 256)", "ReLU_2"),
#     ("ReLU_2", "Linear (256 -> 8, action output)")
# ]

# # Definiert die Schichten für das Critic-Netzwerk
# critic_layers = [
#     ("Input (Flatten)", "Linear (290 -> 512)"),
#     ("Linear (290 -> 512)", "ReLU"),
#     ("ReLU", "Linear (512 -> 256)"),
#     ("Linear (512 -> 256)", "ReLU_2"),
#     ("ReLU_2", "Linear (256 -> 1, Q-value)"),
#     ("ReLU", "Branch"),
#     ("Branch", "Linear (512 -> 256)"),
#     ("Linear (512 -> 256)", "ReLU_3"),
#     ("ReLU_3", "Linear (256 -> 1, Q-value_2)")
# ]

# # Fügt die Kanten für das Actor-Netzwerk hinzu
# G.add_edges_from(actor_layers, label="Actor Network")

# # Fügt die Kanten für das Critic-Netzwerk hinzu
# G.add_edges_from(critic_layers, label="Critic Network")

# # Positioniert die Knoten
# pos = nx.spring_layout(G, seed=42)

# # Zeichne den Graphen neu mit strukturierter Positionierung
# plt.figure(figsize=(12, 6))
# nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', arrows=True)

# # Zeigt den Graphen an
# plt.title("Structured Actor and Critic Network Structure", size=15)
# plt.show()




