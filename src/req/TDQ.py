import numpy as np

class TDQ:
    def __init__(self, map_dim, action_space_len, alpha):
        self.q_func = np.zeros((map_dim, map_dim, action_space_len))
        self.action_space = np.arange(action_space_len)
        self.alpha = alpha

    def q_val(self, state, action):
        return self.q_func[state[0]][state[1]][action]

    def q_val_max_action(self, state):
        q_vals  = self.q_func[state[0]][state[1]]
        actions = np.where(q_vals == np.max(q_vals))
        return actions[0]

    def q_val_max_valid(self, state, valid_actions):
        q_vals  = self.q_func[state[0]][state[1]][valid_actions]
        return np.max(q_vals)

    def max_sample_valid(self, state, valid_actions):
        q_vals = [self.q_val(state, a) for a in valid_actions]
        max_q = max(q_vals)
        max_q_actions = [a for a in valid_actions if self.q_val(state, a) == max_q]
        return np.random.choice(max_q_actions)

    def update(self, target, state, action):
        self.q_func[state[0]][state[1]][action] += self.alpha * (target - self.q_func[state[0]][state[1]][action])