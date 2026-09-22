import numpy as np

class Replay:
    def __init__(self, n_step):
        self.size = n_step
        self.states = []
        self.n_states = []
        self.actions = []
        self.rewards = []
        self.truncated = []
        self.terminated = []

    def clear(self):
        self.states = []
        self.n_states = []
        self.actions = []
        self.rewards = []
        self.truncated = []
        self.terminated = []

    def jotDown(self, state, action, reward, n_state, trunc, term):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.n_states.append(n_state)
        self.truncated.append(trunc)
        self.terminated.append(term)

        if len(self.states) > self.size:
            self.states.pop(0)

        if len(self.actions) > self.size:
            self.actions.pop(0)

        if len(self.rewards) > self.size:
            self.rewards.pop(0)

        if len(self.truncated) > self.size:
            self.truncated.pop(0)
    
        if len(self.terminated) > self.size:
            self.terminated.pop(0)

        if len(self.n_states) > self.size:
            self.n_states.pop(0)

    def pop_oldest(self):
        self.states.pop(0)
        self.actions.pop(0)
        self.rewards.pop(0)
        self.truncated.pop(0)
        self.terminated.pop(0)
        self.n_states.pop(0)