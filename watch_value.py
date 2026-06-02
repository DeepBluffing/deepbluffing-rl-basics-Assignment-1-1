import gymnasium as gym
import time
from value_iteration import ValueIteration
from dp_utils import TabularValueFunction
from visualize_dp import SimpleGridMDP

def watch_dp_agent():
    print("The agent is thinking (Running Value Iteration)...")
    
    # 1. Run the Bellman Math on the perfect model
    mdp = SimpleGridMDP()
    values = TabularValueFunction()
    vi = ValueIteration(mdp, values)
    vi.value_iteration(max_iterations=100)
    
    print("Math complete! The agent has a perfect map in its head.")
    print("Opening the visualizer...")

    # 2. Boot up the Gymnasium GUI
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    state, _ = env.reset()
    done = False

    time.sleep(1) # Wait for window to open

    while not done:
        # 3. Extract the optimal policy directly from the calculated V(s)
        best_action = None
        max_q = float('-inf')
        
        for action in mdp.get_actions(state):
            q_value = 0.0
            for (new_state, prob) in mdp.get_transitions(state, action):
                reward = mdp.get_reward(state, action, new_state)
                # Bellman expectation to find the Q-value of this action
                q_value += prob * (reward + mdp.get_discount_factor() * values.get_value(new_state))
            
            if q_value > max_q:
                max_q = q_value
                best_action = action
        
        # 4. Take the mathematically optimal step
        state, reward, terminated, truncated, _ = env.step(best_action)
        done = terminated or truncated
        
        time.sleep(0.5) # Slow down the frames

    if reward == 1.0:
        print("The Elf reached the treasure using Dynamic Programming!")
    else:
        print("The Elf fell in a hole. Check your Bellman equations!")
        
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    watch_dp_agent()
