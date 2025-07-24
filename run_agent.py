# run_agent.py
import numpy as np
import pickle
import pygame
import time
from key_door_env import KeyDoorEnv

def load_and_run_agent(q_table_path="q_table.pkl", max_steps=100):
    """
    Loads the trained Q-table and runs the agent with visualization
    """
    # Load environment
    env = KeyDoorEnv()
    
    # Load Q-table
    with open(q_table_path, "rb") as f:
        Q_table = pickle.load(f)
    
    # Reset environment
    state, _ = env.reset()
    total_reward = 0
    
    print("Starting Agent Run...")
    print("Controls: Close the pygame window to stop early")
    
    for step in range(max_steps):
        x, y, k = state.astype(int)
        
        # Get best action from Q-table
        action = np.argmax(Q_table[x, y, k])
        
        # Take action
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Render environment
        env.render()
        time.sleep(0.3)  # Control speed here (smaller = faster)
        
        # Print step info
        action_names = ["Up", "Down", "Right", "Left"]
        print(f"Step {step+1}: Pos=({x},{y}), Key={k}, Action={action_names[action]}, Reward={reward:.2f}")
        
        # Check for pygame window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        if done:
            print(f"Episode completed! Total reward: {total_reward:.2f}")
            break
            
        state = next_state
    
    env.close()

# run 
if __name__ == "__main__":
    load_and_run_agent()