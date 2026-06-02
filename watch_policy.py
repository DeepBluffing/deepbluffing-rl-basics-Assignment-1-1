import gymnasium as gym
import time
from policy_iteration import PolicyIteration
from dp_utils import TabularPolicy
from visualize_dp import SimpleGridMDP

def watch_pi_agent():
    print("The agent is writing its rulebook (Running Policy Iteration)...")
    
    # 1. Setup the perfect model
    mdp = SimpleGridMDP()
    
    # Initialize a terrible default policy (e.g., always go Left)
    policy = TabularPolicy(default_action=0) 
    
    # 2. Run Policy Iteration
    pi = PolicyIteration(mdp, policy)
    iterations = pi.policy_iteration(max_iterations=100)
    
    print(f"Rulebook complete in {iterations} iterations!")
    print("Opening the visualizer...")

    # 3. Boot up the Gymnasium GUI
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
    state, _ = env.reset()
    done = False

    time.sleep(1) # Wait for window to open

    while not done:
        # 4. Ask the converged policy for the exact action (No math needed here!)
        actions = mdp.get_actions(state)
        best_action = policy.select_action(state, actions)
        
        # 5. Take the step
        state, reward, terminated, truncated, _ = env.step(best_action)
        done = terminated or truncated
        
        time.sleep(0.5) # Slow down the frames

    if reward == 1.0:
        print("The Elf reached the treasure using Policy Iteration!")
    else:
        print("The Elf fell in a hole. Check your Bellman equations!")
        
    time.sleep(2)
    env.close()

if __name__ == "__main__":
    watch_pi_agent()
