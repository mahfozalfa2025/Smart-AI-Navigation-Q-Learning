import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import time  #  Used to control speed of rendering

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma):
    """
    Trains a Q-learning agent in the given environment.
    It displays the agent's movement visually during training and saves the learned Q-table.
    """

    # State space: (x, y, has_key) → has_key is binary: 0 or 1
    n_states = (env.grid_size, env.grid_size, 2)
    n_actions = env.action_space.n

    # Initialize Q-table
    Q_table = np.zeros(n_states + (n_actions,))
    rewards = []  # To track total reward per episode

    for ep in range(no_episodes):
        state, _ = env.reset()
        total_reward = 0

        print(f"\nEpisode {ep + 1}:")

        for step in range(100):  # Max steps per episode
            x, y, k = state.astype(int)

            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[x, y, k])

            # Apply action and observe result
            next_state, reward, done, _, _ = env.step(action)
            nx, ny, nk = next_state.astype(int)

            # Q-learning update
            Q_table[x, y, k, action] += alpha * (
                reward + gamma * np.max(Q_table[nx, ny, nk]) - Q_table[x, y, k, action]
            )

            state = next_state
            total_reward += reward

            # Visualize movement
            env.render()
            time.sleep(0.01)  #  reduce delay to speed up rendering (can be 0.005 or even 0)

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

        #  Print reward after every episode
        print(f"Total Reward: {total_reward} | Epsilon: {epsilon:.3f}")

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    # Plot reward curve
    plt.plot(rewards)
    plt.title("Total Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

def visualize_q_table_dqn(q_table_path="q_table.pkl",
                          grid_size=5,
                          goal_coordinates=(4, 4),
                          key_coordinates=(2, 2),
                          danger_coordinates=[(1, 1), (3, 3)],
                          actions=["Up", "Down", "Right", "Left"],
                          cmap_style="plasma"):
    """
    Visualizes the Q-table as heatmaps for each action and key state (has_key = 0 or 1).
    Helps understand the learned behavior of the agent for each possible action.
    """

    # Load saved Q-table
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)

    # Create a heatmap for each action in both key states
    for has_key in [0, 1]:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Q-table Heatmaps (has_key = {has_key})", fontsize=16)

        for i, action in enumerate(actions):
            data = q_table[:, :, has_key, i].copy()  # Extract Q-values for the action

            # Create a mask to hide cells with special labels
            mask = np.zeros_like(data, dtype=bool)
            gx, gy = goal_coordinates
            kx, ky = key_coordinates
            mask[gx, gy] = True
            mask[kx, ky] = True
            for dx, dy in danger_coordinates:
                mask[dx, dy] = True

            # Draw heatmap
            ax = axes[i]
            sns.heatmap(data,
                        annot=True,
                        fmt=".2f",
                        cmap=cmap_style,
                        cbar=False,
                        ax=ax,
                        annot_kws={"size": 9},
                        mask=mask)

            # Mark special cells on top of heatmap
            ax.text(gy + 0.5, gx + 0.5, 'G', ha='center', va='center',
                    color='green', fontsize=14, weight='bold')
            ax.text(ky + 0.5, kx + 0.5, 'K', ha='center', va='center',
                    color='blue', fontsize=14, weight='bold')
            for dx, dy in danger_coordinates:
                ax.text(dy + 0.5, dx + 0.5, 'H', ha='center', va='center',
                        color='red', fontsize=14, weight='bold')

            ax.set_title(f"Action: {action}")
            ax.set_xlabel("Y")
            ax.set_ylabel("X")

        plt.tight_layout()
        plt.show()

# ================================
# TEMP TEST CODE
# ================================
if __name__ == "__main__":
    from key_door_env import KeyDoorEnv

    # Create environment
    env = KeyDoorEnv()

    # Try training for just 3 episodes
    train_q_learning(
        env=env,
        no_episodes=1000,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99,
        alpha=0.1,
        gamma=0.9
    )

    # Try visualizing the Q-table after that
    visualize_q_table_dqn(
        q_table_path="q_table.pkl",
        grid_size=5,
        goal_coordinates=(4, 4),
        key_coordinates=(2, 2),
        danger_coordinates=[(1, 1), (3, 3)]
    )

