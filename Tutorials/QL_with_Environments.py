import numpy as np
import pylab as plt
import networkx as nx

# Generating the map and paths for the agent

## Creating points
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]

## Our goal point we want to reach
goal = 7

## Adding environmental obstacles

### Bees
bees = [2]

### Smoke
smoke = [4, 5, 6]

## Graphing the nodes and edges
G = nx.Graph()
G.add_edges_from(points_list)
mapping = {0 : 'Start',
           1 : '1',
           2 : '2 - Bees',
           3 : '3',
           4 : '4 - Smoke',
           5 : '5',
           6 : '6 - Smoke',
           7 : '7 - Beehive'}
H = nx.relabel_nodes(G, mapping)
pos = nx.spring_layout(H)
nx.draw_networkx_nodes(H, pos, node_size=[200, 200, 200, 200, 200, 200, 200, 200])
nx.draw_networkx_edges(H, pos)
nx.draw_networkx_labels(H, pos)
plt.show()

# Generating the reward matrix

## Size of the matrix is the number of points in graph
MATRIX_SIZE = 8

## Creating the matrix
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))

## Filling in matrix with -1 values
R *= -1

## If path is viable change value to 0, if it is a goal path change it to 100, everything else stays -1
for point in points_list:
    print(point)

    ### if point has a 7 in it, then reward is 100, else 0
    if point[1] == goal:
        R[point] = 100
    else:
        R[point] = 0
    
    if point[0] == goal:
        R[point[::-1]] = 100
    else:
        R[point[::-1]] = 0

## If destination is reach, add 100
R[goal,goal] = 100

# Generating the Q-table

## Initializing matrices for each environmental obstacles
enviro_bees = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
enviro_smoke = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

## Initializing Q-table with an empty matrix
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

## seting the learning parameter
gamma = 0.8

## setting the intial state
initial_state = 1

# Functions for actions taken by the agent

## Check for the available actions in a given state
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

## Return a random action given a range of actions
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1))
    return next_action 

## Checks to see if agent runs into obstacles
def collect_environmental_data(action):
    found = []
    if action in bees:
        found.append('b')
    if action in smoke:
        found.append('s')
    return found

## Function to update Q-table with the new information the agent experienced
def update(current_state, action, gamma):
    ### Find the column where the action has the most reward
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    ### If max_index has more than one action, choose a random action
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    
    max_value = Q[action, max_index]

    ### Update the Q-table with the most recent information
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    print('max_value', R[current_state, action] + gamma * max_value)

    environment = collect_environmental_data(action)
    if 'b' in environment:
        enviro_bees[current_state, action] += 1
    
    if 's' in environment:
        enviro_smoke[current_state, action] += 1
    
    if np.max(Q) > 0:
        return np.sum(Q/np.max(Q)*100)
    else:
        return 0

available_act = available_actions(initial_state)
action = sample_next_action(available_act)
update(initial_state, action, gamma)

scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state, action, gamma)


# print environmental matrices
print('Bees Found')
print(enviro_bees)
print('Smoke Found')
print(enviro_smoke)

# Re-initializing 
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

# Creating a single matrix that gives positive value for bees and negative value for smoke
enviro_matrix = enviro_bees - enviro_smoke

## Function to update Q-table with the new information the agent experienced
def new_update(current_state, action, gamma):
    ### Find the column where the action has the most reward
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    ### If max_index has more than one action, choose a random action
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)
    
    max_value = Q[action, max_index]

    ### Update the Q-table with the most recent information
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    print('max_value', R[current_state, action] + gamma * max_value)

    environment = collect_environmental_data(action)
    if 'b' in environment:
        enviro_matrix[current_state, action] += 1
    
    if 's' in environment:
        enviro_matrix[current_state, action] -= 1
    
    if np.max(Q) > 0:
        return np.sum(Q/np.max(Q)*100)
    else:
        return 0

# Get available actions in the current state
available_act = available_actions(initial_state)

# Sample next action to be performed
action = sample_next_action(available_act)

# Updating Q-table with actions taken
new_update(initial_state,action,gamma)

enviro_matrix_snap = enviro_matrix.copy()

# Selects an action that reduces risk of running into smoke
def available_actions_with_enviro_help(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]

    env_pos_row = enviro_matrix_snap[state, av_act]

    if np.sum(env_pos_row < 0):
        temp_av_act = av_act[np.array(env_pos_row)[0]>=0]
        if len(temp_av_act) > 0:
            print("Going from:", av_act)
            print("to: ", temp_av_act)
            av_act = temp_av_act
    return av_act

# Training 
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions_with_enviro_help(current_state)
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)
    print ('Score:', str(score))

## Starting from 0
current_state = 0

## List to collect the optimized path agent discovered
steps = [current_state]

## Run algorithm until agent reaches goal
while current_state != goal:

    ### Get the highest reward action from the Q-table
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state,]))[1]

    ### If there are multiple actions, randomly choice one, else choose that action
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    
    ### Add the action taken in the steps list
    steps.append(next_step_index)

    ### Update the current state
    current_state = next_step_index

print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show()