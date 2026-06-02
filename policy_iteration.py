from dp_utils import TabularPolicy, TabularValueFunction, QTable

class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001):
        """ Evaluates the given policy until values converge within theta. """
        while True:
            delta = 0.0
            new_values = TabularValueFunction()
            
            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                old_value = values.get_value(state)
                
                # ==========================================
                # TODO 1: Policy Evaluation Update
                # Use policy.select_action(state, actions) to find the current policy's choice.
                # Calculate the expected return (Q-value) of taking ONLY that action.
                # Save this expected return into 'new_values'.
                # ==========================================
                
                pass # Replace this pass with your logic
                
            self.values.merge(new_values)
            if delta < theta:
                break
        return values

    def policy_iteration(self, max_iterations=100, theta=0.001):
        """ Implements policy iteration. Returns the number of iterations executed. """
        values = TabularValueFunction()

        for i in range(1, max_iterations + 1):
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta)
            
            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                old_action = self.policy.select_action(state, actions)

                # ==========================================
                # TODO 2: Policy Improvement Update
                # Calculate the Q-value for EVERY possible action in this state.
                # Find the action with the maximum Q-value.
                # Update self.policy to choose this new best action.
                # ==========================================
                
                new_action = None # Update this variable
                
                if new_action is not old_action:
                    policy_changed = True

            if not policy_changed:
                return i

        return max_iterations
