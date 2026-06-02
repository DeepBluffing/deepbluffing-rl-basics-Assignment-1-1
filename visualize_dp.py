from value_iteration import ValueIteration
from dp_utils import TabularValueFunction, visualize_policy_ascii

# ==========================================
# 1. DEFINE THE ENVIRONMENT (4x4 Gridworld)
# ==========================================
class SimpleGridMDP:
    def get_states(self):
        return list(range(16)) # States 0 through 15

    def get_actions(self, state):
        # States 5, 7, 11, 12 are "Holes". State 15 is the "Treasure".
        # If we are in a hole or at the treasure, the game is over (no actions).
        if state in [5, 7, 11, 12, 15]: 
            return [] 
        return [0, 1, 2, 3] # 0:Left, 1:Down, 2:Right, 3:Up

    def get_transitions(self, state, action):
        # Calculate the next state based on the action
        row = state // 4
        col = state % 4
        
        if action == 0:   col = max(0, col - 1) # Left
        elif action == 1: row = min(3, row + 1) # Down
        elif action == 2: col = min(3, col + 1) # Right
        elif action == 3: row = max(0, row - 1) # Up
        
        next_state = row * 4 + col
        return [(next_state, 1.0)] # 100% chance of moving to that square

    def get_reward(self, state, action, next_state):
        if next_state == 15: return 10.0 # Huge reward for the treasure!
        if next_state in [5, 7, 11, 12]: return -5.0 # Penalty for falling in a hole
        return -0.1 # Tiny penalty for taking a step (encourages the shortest path)

    def get_discount_factor(self):
        return 0.9

# ==========================================
# 2. RUN THE ALGORITHM AND VISUALIZE
# ==========================================
if __name__ == "__main__":
    print("🤖 Building the Gridworld...")
    mdp = SimpleGridMDP()
    values = TabularValueFunction()
    
    print("🧠 Running Value Iteration...")
    vi = ValueIteration(mdp, values)
    iterations = vi.value_iteration(max_iterations=100)
    
    print(f"✅ Algorithm converged in {iterations} iterations!")
    
    # Draw the resulting policy map!
    visualize_policy_ascii(mdp, values, grid_size=4, goal_state=15)
