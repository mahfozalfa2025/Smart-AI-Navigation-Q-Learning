import sys
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class KeyDoorEnv(gym.Env):
    """
    A custom Grid-World Gymnasium environment with:
    - A player (agent)
    - A key that must be picked up
    - A locked goal (door) that can only be accessed if key is picked
    - Dangerous cells that end the episode with negative reward
    """

    def __init__(self, grid_size=5):
        super().__init__()

        # Grid and cell configuration
        self.grid_size = grid_size
        self.cell_size = 100 #pixel

        # Define action space: 0=Up, 1=Down, 2=Right, 3=Left
        self.action_space = spaces.Discrete(4)

        # Observation space includes (x, y, has_key) => (row, column, binary key flag)
        self.observation_space = spaces.MultiDiscrete([grid_size, grid_size, 2])

        # Initialize PyGame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((grid_size * self.cell_size, grid_size * self.cell_size))
        pygame.display.set_caption("Key-Door-Danger Environment")

        # Load and scale images for different entities
        self.agent_img = pygame.transform.smoothscale(pygame.image.load("agent.png"), (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.smoothscale(pygame.image.load("goal.png"), (self.cell_size, self.cell_size))
        self.key_img = pygame.transform.smoothscale(pygame.image.load("key.png"), (self.cell_size, self.cell_size))
        self.danger_img = pygame.transform.smoothscale(pygame.image.load("danger.png"), (self.cell_size, self.cell_size))

        # Reset environment to initial state
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state:
        - Agent at top-left
        - Key and goal at fixed positions
        - Dangers at predefined positions
        - Agent doesn't have the key initially
        """
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([4, 4])
        self.key_pos = np.array([2, 2])
        self.danger_pos = [np.array([1, 1]), np.array([3, 3])]
        self.has_key = 0
        self.done = False

        obs = np.append(self.agent_pos, self.has_key)
        return obs, {}

    def step(self, action):
        """
        Executes a step in the environment based on the action:
        - Moves agent
        - Picks up key if on key tile
        - Checks for danger or reaching the goal
        - Applies rewards/penalties
        """
        if self.done:
            return np.append(self.agent_pos, self.has_key), 0, True, False, {}

        # Move the agent
        if action == 0 and self.agent_pos[0] > 0:            # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] < self.grid_size - 1:  # Right
            self.agent_pos[1] += 1
        elif action == 3 and self.agent_pos[1] > 0:           # Left
            self.agent_pos[1] -= 1

        # Check if key is picked up
        if np.array_equal(self.agent_pos, self.key_pos):
            self.has_key = 1

        # If agent steps into danger => terminate with large negative reward
        if any(np.array_equal(self.agent_pos, d) for d in self.danger_pos):
            return np.append(self.agent_pos, self.has_key), -20, True, False, {}

        reward = -0.01  # Small step penalty to encourage faster solution

        # Check if goal is reached
        if np.array_equal(self.agent_pos, self.goal_pos):
            if self.has_key:
                reward = 10
                self.done = True
            else:
                reward = -1  # Penalty for reaching door without key

        obs = np.append(self.agent_pos, self.has_key)
        return obs, reward, self.done, False, {}

    def render(self):
        """
        Visualizes the environment using PyGame:
        - Draws grid
        - Places key, goal, danger cells, and agent
        """
        for event in pygame.event.get(): #close the window (bckg)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))  # White background

        # Draw grid lines
        for x in range(self.grid_size): # x lines 
            for y in range(self.grid_size): # y lines
                pygame.draw.rect(self.screen, (200, 200, 200), #gray color
                                 (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size), 1)

        # Draw entities
        if not self.has_key: # flip condition to show key or not
            self.screen.blit(self.key_img, (self.key_pos[1]*self.cell_size, self.key_pos[0]*self.cell_size))
        self.screen.blit(self.goal_img, (self.goal_pos[1]*self.cell_size, self.goal_pos[0]*self.cell_size))
        for danger in self.danger_pos:
            self.screen.blit(self.danger_img, (danger[1]*self.cell_size, danger[0]*self.cell_size))
        self.screen.blit(self.agent_img, (self.agent_pos[1]*self.cell_size, self.agent_pos[0]*self.cell_size))

        pygame.display.flip() # update grid
        pygame.time.wait(1) # dispaly speed control

    def close(self):
        pygame.quit()


# ------------------ Test Run ------------------
if __name__ == "__main__":
    env = KeyDoorEnv()
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample() # random movement
        obs, reward, done, _, _ = env.step(action) # output
        env.render() # visual
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            break
    env.close()
