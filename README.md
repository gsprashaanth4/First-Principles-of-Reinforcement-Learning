## Tabular Reinforcement Learning Algorithms

This is an educational project, implementing classical reinforcement learning algorithms from first principles without using neural networks. The implementations use tabular state-action value functions and a custom GridWorld environment. The program files include implementations of:

- Monte Carlo control
- n-step Temporal-Difference Q-learning
- tabular Q-value functions
- epsilon-greedy action selection
- valid-action filtering
- discounted return estimation
- bootstrapped TD targets
- experience replay for n-step transitions
- decaying epsilon exploration
- configurable learning rate and discount factor
- reward shaping
- Pygame based environment visualization
- training reward visualization using Matplotlib

### Algorithms

Monte Carlo:
- Collects complete episode trajectories before updating the Q-function.
- Calculates discounted returns by traversing the trajectory backwards.

n-step TD-Q:
- Collects n-step transitions using a replay buffer.
- Calculates multi-step returns with bootstrapping from the estimated value of the next state.
- Handles terminal and truncated episodes while processing remaining transitions - transition flush mech.

### Demo

Both algorithms are trained on custom GridWorld environments with obstacles, terminal states and discrete movement actions.

Monte Carlo training:
![MC](https://github.com/gsprashaanth4/First-Principles-of-Reinforcement-Learning/blob/main/media/monte.gif)

n-step TD-Q training:
![TD-Q](https://github.com/gsprashaanth4/First-Principles-of-Reinforcement-Learning/blob/main/media/TDQ.gif)

The project is primarily intended as a proof of work and learning implementation of classical tabular reinforcement learning methods.