import numpy as np

# The dimensions of the warehouse is 11x11
dim = 11

# The actions for each state
actions = ["up", "down", "left", "right"]

# The Q-table

## Creating a Q-table with dimensions of  (11 x 11 x 4) (11 x 11 for each state in environment and 4 for number of actions in each state)
Q_table = np.zeros((dim, dim, len(actions)))

# The Reward System

## Creating a reward system with all values of -100
rewards = np.full((dim, dim), -100.)

## Setting the shipping dock (goal) as 100
rewards[0,5] = 100

## Setting up the aisles (possible locations an agent can go through)
aisles = {}
aisles[1] = [i for i in range (1,10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1,8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1,10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

## Setting the reward for each aisle (possible locations) value as -1 
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1

## print rewards matrix
for row in rewards:
    print(row)

# TRAINING

## Functions

### Determines if the current state is the terminal state
def is_terminal_state(current_row_index, current_column_index):
    
    # Return false if the reward is -1 for a state (it is NOT a terminal state because terminal state has reward of 100)
    if rewards[current_row_index, current_column_index]  == -1:
        return False
   
   # Return true otherwise
    else:
        return True

### Randomly select a starting location for an agent
def get_start_location():
    
    # Randomly choosing a location (row, column of warehouse) of the agent
    current_row_index = np.random.randint(dim)
    current_column_index = np.random.randint(dim)

    # If random location is terminal state, choose another random location
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(dim)
        current_column_index = np.random.randint(dim)
    
    # Return the random location (row, column)
    return current_row_index, current_column_index

### Function that chooses an action
def get_next_action(current_row_index, current_column_index, epsilon):
    
    # Provide a high probability to take the greedy option (highest reward)
    if np.random.random() < epsilon:
        return np.argmax(Q_table[current_row_index, current_column_index])
    
    # Else randomly select an action (provides ability to explore other actions)
    else: 
        return np.random.randint(4)


### Get the next location (row, column) after an action was chosen
def get_next_location(current_row_index, current_column_index, action_index):
    
    new_row_index = current_row_index
    new_column_index = current_column_index

    # If the action was up, the row index decreases by 1 while column index stays the same
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    
    # If the action was right, the column index increases by 1 while row index stays the same
    elif actions[action_index] == 'right' and current_column_index < dim - 1:
        new_column_index += 1
    
    # If the action was down, the row index increases by 1 while column index stays the same
    elif actions[action_index] == 'down' and current_row_index < dim - 1:
        new_row_index += 1 
    
    # If the action was left, the column index decreases by 1 while row index stays the same
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    
    # Return the new location (row, column)
    return new_row_index, new_column_index

### Get the shortest path given the start location (row, column)
def get_shortest_path(start_row_index, start_column_index):
    
    # Return empty path if start location is already at the terminal state
    if is_terminal_state(start_row_index, start_column_index):
        return []
    
    # Else find the shortest path
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        
        # Create a list to collect the locations to generate a path
        shortest_path = []
        
        # Add the current location to the path
        shortest_path.append([current_row_index, current_column_index])

        # While the current location is NOT the terminal state, take an action -> get to the new location -> add the location into the path
        while not is_terminal_state(current_row_index, current_column_index):
            
            # Gets the next action
            action_index = get_next_action(current_row_index, current_column_index, 1.)

            # Gets the location after the action is taken
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)

            # Add the new location to the path
            shortest_path.append([current_row_index, current_column_index])
        
        # Return the path 
        return shortest_path

## Training parameters

### Chance that the highest reward is taken
epsilon = 0.9

### How much should future rewards be considered for current decision of action
discount_factor = 0.9

### Rate of how much the agent should learn
learning_rate = 0.9

print("TRAINING!")
### Training the agent for 1000 iterations to fill up the Q-table
for episode in range(1000):

    # Getting the start location of 
    row_index, column_index = get_start_location()

    while not is_terminal_state(row_index, column_index):
        
        # Gets the action 
        action_index = get_next_action(row_index, column_index, epsilon)

        # Saves the old row and index
        old_row_index, old_column_index = row_index, column_index
        
        # Gets the new state by taking that action
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        # Gets the current reward 
        reward = rewards[row_index, column_index]

        # Gets the current Q-value of the state
        old_q_value = Q_table[old_row_index, old_column_index, action_index]

        # Calculates the temporal difference given the reward, discount factor, the next Q-value after the action is taken, and the current Q-value
        temporal_difference = reward + (discount_factor * np.max(Q_table[row_index, column_index]) - old_q_value)

        # Updates the current Q-value with the new one
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        Q_table[old_row_index, old_column_index, action_index] = new_q_value

print("COMPLETE!")

# Finding the shortest paths given three different starting positions

## Starting location (7,3)
print("Starting Location (7,3): ")
print(get_shortest_path(7, 3))

## Starting location (5,0)
print("Starting Location (5,0): ")
print(get_shortest_path(5, 0))

## Starting location (9,7)
print("Starting Location (9,7): ")
print(get_shortest_path(9, 7))