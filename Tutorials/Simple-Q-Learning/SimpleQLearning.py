import numpy as np
import pylab as plt
import networkx as nx

# Mapping cell to cell, with a circular cell for goal point #

## Creating points
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]

## Our goal point we want to reach
goal = 7

## Creating the network map layout
G = nx.Graph()
G.add_edges_from(points_list)

## Randomly generating the paths from each point
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()

# Generating the Reward Matrix #

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

## If destination is reached, add 100
R[goal,goal] = 100

# Generating the Q-table #

## Creating an empty matrix for Q-table
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

## seting the learning parameter
gamma = 0.8

## setting the intial state
initial_state = 1

## Creating a function that see the possible actions agent can take
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

## Getting a random action from a range of actions
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_actions_range, 1))
    return next_action

## Updating the Q-table 
def update(current_state, action, gamma):
    ### find all the indices of the max values in the row
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]

    ### if there are more than one index, randomly choice an action, else take that action
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    ### get the value from Q-table given the max_index
    max_value = Q[action, max_index]

    ### Updating the reward for an action taken in the current state
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    print('max value', R[current_state, action] + gamma * max_value)
    
    if np.max(Q) > 0:
        return np.sum(Q/np.max(Q)*100)
    else:
        return 0

# Training 

## list of scores collected from each training session
scores = []

## Training for 700 sessions
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state, action, gamma)
    scores.append(score)
    print('Score', str(score))

print("Trained Q matrix: ")
print(Q/np.max(Q)*100)

# Testing

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