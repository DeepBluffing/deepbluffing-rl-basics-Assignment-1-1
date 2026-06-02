class TabularValueFunction:
    """ Stores the value V(s) for each state. """
    def __init__(self):
        self.values = {}

    def get_value(self, state):
        return self.values.get(state, 0.0) # Default value is 0.0

    def add(self, state, value):
        self.values[state] = value

    def merge(self, other):
        self.values.update(other.values)


class QTable:
    """ Stores the action-value Q(s, a) for each state-action pair. """
    def __init__(self, alpha=1.0):
        self.q_values = {}
        self.alpha = alpha

    def get_q(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def update(self, state, action, value):
        self.q_values[(state, action)] = value

    def get_max_q(self, state, actions):
        if not actions:
            return 0.0
        return max(self.get_q(state, a) for a in actions)

    def get_argmax_q(self, state, actions):
        if not actions:
            return None
        
        best_action = actions[0]
        best_q = self.get_q(state, best_action)
        
        for a in actions[1:]:
            q = self.get_q(state, a)
            if q > best_q:
                best_q = q
                best_action = a
        return best_action


class TabularPolicy:
    """ Stores the deterministic policy pi(s) -> a. """
    def __init__(self, default_action=0):
        self.policy_map = {}
        self.default_action = default_action

    def select_action(self, state, actions):
        # Return the learned action if it exists
        if state in self.policy_map:
            return self.policy_map[state]
        # Otherwise, return the default fallback action
        if actions and self.default_action in actions:
            return self.default_action
        if actions:
            return actions[0]
        return None

    def update(self, state, action):
        self.policy_map[state] = action


def visualize_policy_ascii(mdp, values, grid_size=4, goal_state=15):
    """
    Prints a visual representation of the optimal policy using arrows.
    Actions: 0=Left, 1=Down, 2=Right, 3=Up
    """
    print("\n🗺️  Optimal Policy Map  🗺️")
    print("-" * (grid_size * 6))
    
    for row in range(grid_size):
        row_string = "|"
        for col in range(grid_size):
            state = row * grid_size + col
            
            if state == goal_state:
                row_string += " 💎 |"
                continue
            
            best_action = None
            max_q = float('-inf')
            
            for action in mdp.get_actions(state):
                q_value = 0.0
                for (new_state, prob) in mdp.get_transitions(state, action):
                    reward = mdp.get_reward(state, action, new_state)
                    q_value += prob * (reward + mdp.get_discount_factor() * values.get_value(new_state))
                
                if q_value > max_q:
                    max_q = q_value
                    best_action = action
            
            if best_action == 0:   row_string += "  ← |"
            elif best_action == 1: row_string += "  ↓ |"
            elif best_action == 2: row_string += "  → |"
            elif best_action == 3: row_string += "  ↑ |"
            else:                  row_string += "  ? |"
            
        print(row_string)
        print("-" * (grid_size * 6))
