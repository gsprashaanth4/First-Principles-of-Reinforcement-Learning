import numpy as np
from req.Visuals import Visuals

class GridWorld:
    def __init__(self):
        self.state = np.zeros(2, dtype=int)

        self.map_dim = 6
        self.map = np.array(([0,0,0,0,0,0],
                             [0,0,1,1,1,2],
                             [0,0,0,0,1,1],
                             [0,0,0,0,0,0],
                             [1,1,1,1,0,0],
                             [0,0,0,0,0,0]))

        self.action_space = np.array([0,1,2,3])
        self.action_space_len = 4
        self.max_steps = 25

        self.vis = Visuals(self.map, self.map_dim)
        
        self.step_r = -5
        self.goal_r = 10
        self.penl_r = -2
        self.invl_r = -9

        self.prev_state = None
        self.prev_action = None

        self.terminated = False
        self.truncated = False
        self.itr = 0

    def reset(self):
        self.state = np.array([self.map_dim-1, 0])

        self.prev_state = None
        self.prev_action = None

        self.terminated = False
        self.truncated = False
        self.itr = 0
        self.vis.reset()

        self.vis.update(self.state, self.itr)

        return np.array(self.state), self.terminated, self.truncated

    def get_valid_action(self, state=None):

        if state is None:
            state = self.state

        valid_actions = []

        x = self.state[0]
        y = self.state[1]
        if x != 0              and self.map[x-1][y] != 1: valid_actions.append(0)
        if y != self.map_dim-1 and self.map[x][y+1] != 1: valid_actions.append(1)
        if x != self.map_dim-1 and self.map[x+1][y] != 1: valid_actions.append(2)
        if y != 0              and self.map[x][y-1] != 1: valid_actions.append(3)
        
        return valid_actions

    def sample(self):
        valid_actions = self.get_valid_action()
        action = np.random.choice(valid_actions)
        return action

    def step(self, action, q_func):
        valid_actions = self.get_valid_action()
        x = self.state[0]
        y = self.state[1]
        reward = 0

        if action in valid_actions:
            if   action == 0: x-=1
            elif action == 1: y+=1
            elif action == 2: x+=1
            elif action == 3: y-=1

            map_val = self.map[x][y]
            if map_val != 2:
                reward += self.step_r
            elif map_val == 2:
                reward += self.goal_r
                self.terminated = True

            if self.prev_action != None and action != self.prev_action:
                reward += self.penl_r
        else:
            reward += self.invl_r

        self.state = np.array([x, y])

        self.itr += 1
        if (self.itr == self.max_steps):
            self.truncated = True

        self.vis.update(self.state, self.itr)
        print(self.itr)
        self.vis.display(q_func)
        return np.array(self.state), reward, self.terminated, self.truncated