import numpy as np

from req.GridWorld import GridWorld as gw
from req.TDQ import TDQ
from req.Replay import Replay

episodes = 500
alpha = 0.1
discount = 0.95
epsilon = 1.0
n_steps = 4

env = gw()
td_q = TDQ(env.map_dim, env.action_space_len, alpha)
replay = Replay(n_steps)

for ep in range(episodes):
    replay.clear()
    epsilon = max(0, epsilon*0.995)
    state, terminated, truncated = env.reset()


    while not (terminated or truncated):
        print(ep, end=" ")
        if np.random.random() < epsilon: action = env.sample()
        else:
            valid_actions = env.get_valid_action() 
            action = td_q.max_sample_valid(state, valid_actions)

        prev_state = state
        state, reward, terminated, truncated = env.step(action, td_q.q_func)

        #               s_t         a_t    r_t+1  n_state  trunc_t    term_t
        replay.jotDown(prev_state, action, reward, state, truncated, terminated)
        env.prev_action = action
        
        if len(replay.states) == n_steps:
            rev_rewards = reversed(replay.rewards)
            
            if replay.terminated[-1]:
                target = 0.0
            else:
                valid_actions = env.get_valid_action()
                max_q = td_q.q_val_max_valid(replay.n_states[-1], valid_actions)
                target = max_q

            s = replay.states[0]
            a = replay.actions[0]
            
            for r in rev_rewards:
                target = r + discount * target

            td_q.update(target, s, a)
            replay.pop_oldest()

    while replay.states:
        if replay.terminated[-1]:
            target = 0.0
        else:
            valid_actions = env.get_valid_action()
            target = td_q.q_val_max_valid(replay.n_states[-1], valid_actions)
        s, a = replay.states[0], replay.actions[0]
        for r in reversed(replay.rewards):
            target = r + discount * target
        td_q.update(target, s, a)
        replay.pop_oldest()

env.vis.wait_until_closed()