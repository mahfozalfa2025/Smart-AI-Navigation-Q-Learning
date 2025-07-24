# main.py

# -----------------------
# Imports
# -----------------------
from key_door_env import KeyDoorEnv
from Q_learning import train_q_learning, visualize_q_table_dqn

# -----------------------
# User Parameters
# -----------------------
train = True                  # Set to True to train the agent
visualize_results = True      # Set to True to show heatmaps after training

# Training hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
no_episodes = 1000          # Number of episodes for full training

# Environment settings
goal_coordinates = (4, 4)
key_coordinates = (2, 2)
danger_coordinates = [(1, 1), (3, 3)]

# -----------------------
# Execution
# -----------------------

if __name__ == "__main__":
    # Create environment instance
    env = KeyDoorEnv()

    # Train Q-learning agent
    if train:
        train_q_learning(
            env=env,
            no_episodes=no_episodes,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            alpha=alpha,
            gamma=gamma
        )

    # Visualize Q-table heatmaps
    if visualize_results:
        visualize_q_table_dqn(
            q_table_path="q_table.pkl",
            grid_size=env.grid_size,
            goal_coordinates=goal_coordinates,
            key_coordinates=key_coordinates,
            danger_coordinates=danger_coordinates
        )

