# The below code is an implementation of The Monte Carlo method
# Which is a broader subclass of simplistic RL techniques

import pygame
import numpy as np
from req.texts import TextBox
import matplotlib.pyplot as plt

# the environment map, for representative purposes, this will be a model-free implementation
map_dim = 5
map = np.array(([0,0,0,0,0],
                [0,0,0,0,0],
                [0,0,1,0,0],
                [0,0,1,0,0],
                [0,0,1,0,2]))


# pygame initializations
pygame.init()
scale = 80
windo = map_dim*scale
screen = pygame.display.set_mode((windo*2,windo+scale)) # The screen component
pygame.display.set_caption("Monte!!!") # The screen title


# depicting the storage of action space of a given state
actions = np.zeros((map_dim, map_dim))

# initiating the action-value function Qπ(s,a)
q_func = np.zeros((map_dim,map_dim,4))

# Training parameters
action_space = [0,1,2,3]    # set of all actions (discrete action space) 0=↑, 1=→, 2=↓, 3=←
max_steps = 30              # the maximum number of steps per episode
episodes = 400              # Number of episodes to train in
epsilon = 1.0               # initiating the epsilon to 1.0 (choosing 100% random at start)
discount = 0.8              # discount rate (Gamma in general notation)
total_reward = 0.0          # variable to store total un-discounted reward per episode
alpha = 0.1                 # learning rate / updation rate

# reward shaping
normal_step_reward = -1     # regular step reward when not reaching terminal state
goal_reached_reward = 5     # reward for when reaching the terminal state

# starting positions
a = map_dim-1
b = 0

# coordinate variables for plotting
episode_points = np.zeros((episodes))
for i in range(episodes):
    episode_points[i] = i

rewards_points = []

# array for action display (Pygame)
txts = []
path = []

# initiating the displays
for i in range(map_dim):
    for j in range(map_dim):
        color = (255,255,255)
        locX = (i*scale)+windo
        locY = j*scale
        # initiating the display elements
        txts.append(TextBox(locX, locY, scale , scale, color, screen))

for i in range(map_dim):
    for j in range(map_dim):
        color = (0,0,0)
        locX = (i*scale)
        locY = j*scale
        # initiating the display elements
        path.append(TextBox(locX, locY, scale, scale, color, screen))

path.append(TextBox(0, map_dim*scale, map_dim*2*scale, scale, (0,255,0), screen))

# Helper function to draw grid
def draw_grid():
    for i in range(0,map_dim):
        pygame.draw.line(screen, (55, 55, 55), 
             [0, i*scale], 
             [windo*2, i*scale], 1)
        
    for i in range(0,map_dim*2):
        pygame.draw.line(screen, (55, 55, 55), 
             [i*scale, 0], 
             [i*scale, windo], 1)

# drawing (coloring) the blocks 1 and 2
def draw_blocks_final():
    for i in range(map_dim):
        for j in range(map_dim):
            if map[i][j] == 1:
                path[j*map_dim+i].display("x", 20, (75, 20, 20))
            elif map[i][j] == 2 or map[i][j] == 3:
                path[j*map_dim+i].display("final", 20, (75, 75, 20))

# resets map visually turning map = 0 colors and map = 2 colors
def reset_path():
    for i in range(map_dim):
        for j in range(map_dim):
            if map[i][j] == 0:
                color = (0,0,0)
                path[j*map_dim+i].display("", 20, color)
            elif map[i][j] == 2:
                path[j*map_dim+i].display("final", 20, (75, 75, 20))
    
# updating pygame display object and handling quit function
def update_pygame():
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# getting the arrows done and updating txt array
def update_Q_display():
    for ac in range(len(q_func)):
        for bc in range(len(q_func[ac])):
    
            if map[ac][bc] == 0:
                
                color = (20, 75, 20)
                valid_actions = get_valid_actions(ac,bc)

                max_q = max(q_func[ac][bc][x] for x in valid_actions)
                best = [x for x in valid_actions if q_func[ac][bc][x] == max_q]
                action_ch = np.random.choice(best)

                if action_ch == 0:
                    message = "↑" 
                elif action_ch == 1:
                    message = "→" 
                elif action_ch == 2:
                    message = "↓" 
                elif action_ch == 3:
                    message = "←"
    
            elif map[ac][bc] == 2 or map[ac][bc] == 3:
                color = (75, 75, 20)
                message = "final"
    
            elif map[ac][bc] == 1:
                color = (75, 20, 20)
                message = "x"

            txts[bc*map_dim+ac].display(message, 20, color)

# getting valid actions of current state and retuned as valid_actions array
def get_valid_actions(a,b):
    valid_actions = [] # array to store the valid actions available to the agent in the current state
    if a != 0         and map[a-1][b] != 1: valid_actions.append(0)
    if b != map_dim-1 and map[a][b+1] != 1: valid_actions.append(1)
    if a != map_dim-1 and map[a+1][b] != 1: valid_actions.append(2)
    if b != 0         and map[a][b-1] != 1: valid_actions.append(3)

    return valid_actions

# Main Driver
if __name__ == "__main__":

    draw_blocks_final()

    reward_visit_count = 0

    # Main looop for the episodes
    for ep in range(episodes):
        epsilon = max(0.05, epsilon*0.995)      # reducing ε based on episode count
        step = 0                                # step initiated to 0 for the current episode
        status = "continues"                    # status variable denoting the episode's status (log)
        reward = -1                             # reward variable
        message = "*"                           # (log)
        color = (0,255,0)                       # color for the 
        path_color=40                           # path color set
        trajectory = []
        terminated = False

        # looping through the steps of the episode
        while (step < max_steps) and (terminated == False):
            
            comm = "*" # noting the action choosing chance * for random and Q for Function (log)

            valid_actions = get_valid_actions(a,b) # getting all valid actions for current state

            # inducing randomness based on the episode count
            if np.random.random() < epsilon:
                comm = "*" # noting...(log)
                action = np.random.choice(valid_actions) # choosing random valid action
                actions[a][b] = action # storing the action selected for depiction
            else:
                comm = "Q" # noting...(log)

                # choosing the action of the highest return from the Q function
                max_q = max(q_func[a][b][x] for x in valid_actions)
                best = [x for x in valid_actions if q_func[a][b][x] == max_q]
                action = np.random.choice(best)

                actions[a][b] = action # storing the action selected for depiction

            # storing the current state for jrajectories
            prev_a = a
            prev_b = b

            action_c = "↑" # variables used for printing status (log)

            # action symbol (log)
            if action == 0:
                action_c = "↑" # (log)
                a-=1
            elif action == 1:
                action_c = "→" # (log)
                b+=1
            elif action == 2:
                action_c = "↓"
                a+=1
            elif action == 3:
                action_c = "←"
                b-=1

            if map[a][b] != 2:
                reward = normal_step_reward
            elif map[a][b] == 2:
                reward = goal_reached_reward
                terminated = True
                status = "terminates"
                path[b*map_dim + a].display("final", 20, (200,200,0))
                reward_visit_count += 1

            path_color += 5 # incrementing path color - darker
            path[prev_b*map_dim + prev_a].display(str(step), 20, (0,path_color,path_color))

            total_reward+=reward
            # console message readying
            cons_mesg = str(ep) + " in " + str(step) + "s on " + str(list([prev_a, prev_b])) + " chose and did " + comm + "~" + action_c + " gone to " + str(list([a, b])) + " with " + str(total_reward) + " and " + status
            # screen message readying
            scr_mesg = f"episode: {ep+1}  ,  epsilon: {epsilon:.2f}  ,  reward-visit count: {reward_visit_count}"
            
            # adding trajectories for each step
            trajectory.append(list([prev_a, prev_b, action, reward, 0]))
            print(cons_mesg)

            path[-1].display(scr_mesg, 30, (50,50,50)) # printing the screen message

            draw_grid()     # drawing the grid
            update_pygame() # updating pygame object

            step+=1         # incrementing step

        # updation of q_func
        step -=1                                            # decrementing step for clarity in program
        traj_run = step                                     # assigning traj_run loop variable to step
        
        # looping for updation
        while traj_run >=0:
            if traj_run == step: # last trajectory
                trajectory[traj_run][-1] = trajectory[traj_run][-2]                                                 # accu_rewards = reward
            else:
                trajectory[traj_run][-1] = (trajectory[traj_run+1][-1] * discount) + trajectory[traj_run][-2]     # accu_rewards = next_reward * discount + reward

            a_point = int(trajectory[traj_run][0])
            b_point = int(trajectory[traj_run][1])
            action  = int(trajectory[traj_run][2])

            target = trajectory[traj_run][-1]
            q_func[a_point][b_point][action] += (target - q_func[a_point][b_point][action]) * alpha

            traj_run-=1


        # storing the tota reward for visualization
        rewards_points.append(total_reward)

        # Restarting Training parameters
        terminated = False # restarting termination flag
        total_reward = 0.0 # variable to store total un-discounted reward per episode
        
        # resetting starting positions
        a = map_dim-1
        b = 0
        
        print()

        update_Q_display()
        draw_grid()
        update_pygame()
        
        reset_path()

    plt.plot(episode_points, np.array(rewards_points))
    plt.show()
    pygame.quit()