import sys
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration
from dp_utils import TabularValueFunction, TabularPolicy

# ==========================================
# 1. THE MOCK MDP (A tiny 3-state game)
# State 0: Start | State 1: Safe Path | State 2: Treasure (Terminal)
# ==========================================
class MiniTreasureMDP:
    def get_states(self):
        return [0, 1, 2]

    def get_actions(self, state):
        if state == 2: return [] # Terminal state has no actions
        return [0, 1]            # 0 = Safe path, 1 = Risky direct path

    def get_transitions(self, state, action):
        if state == 0 and action == 0: return [(1, 1.0)]
        if state == 0 and action == 1: return [(2, 1.0)]
        if state == 1: return [(2, 1.0)] 
        return []

    def get_reward(self, state, action, new_state):
        if new_state == 2: return 10.0
        return -1.0

    def get_discount_factor(self):
        return 0.9

# ==========================================
# 2. AUTOGRADING VALUE ITERATION
# ==========================================
def test_value_iteration():
    print("\nRunning Autograder for Value Iteration...")
    mdp = MiniTreasureMDP()
    values = TabularValueFunction()
    
    try:
        vi = ValueIteration(mdp, values)
        iterations = vi.value_iteration(max_iterations=50)
    except Exception as e:
        print(f"❌ Error during Value Iteration: {e}")
        return False

    # Mathematical Truth: V(0) should be exactly 10.0
    v_0 = values.get_value(0)
    
    if abs(v_0 - 10.0) < 0.01:
        print(f"Value Iteration Passed! (Converged in {iterations} steps)")
        return True
    else:
        print(f"Value Iteration Failed! Expected V(0)=10.0, got V(0)={v_0}")
        return False

# ==========================================
# 3. AUTOGRADING POLICY ITERATION
# ==========================================
def test_policy_iteration():
    print("\n🤖 Running Autograder for Policy Iteration...")
    mdp = MiniTreasureMDP()
    policy = TabularPolicy(default_action=0) 
    
    try:
        pi = PolicyIteration(mdp, policy)
        iterations = pi.policy_iteration(max_iterations=50)
    except Exception as e:
        print(f"Error during Policy Iteration: {e}")
        return False

    # Mathematical Truth: The optimal action from State 0 is Action 1
    optimal_action = policy.select_action(0, [0, 1])
    
    if optimal_action == 1:
        print(f"Policy Iteration Passed! (Converged in {iterations} steps)")
        return True
    else:
        print(f"Policy Iteration Failed! Expected Action 1 for State 0, got {optimal_action}")
        return False

# ==========================================
# 4. EXECUTE TESTS
# ==========================================
if __name__ == "__main__":
    vi_passed = test_value_iteration()
    pi_passed = test_policy_iteration()

    if vi_passed and pi_passed:
        print("\nSUCCESS! All Dynamic Programming tests passed.")
        sys.exit(0)
    else:
        print("\nFAILURE! Please review your Bellman equations.")
        sys.exit(1)
