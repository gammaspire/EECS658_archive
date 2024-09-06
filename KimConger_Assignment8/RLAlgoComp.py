'''
EECS-658 Assignment 8

Part 1: Monte Carlo Algorithm
• Write a Python program that uses the MC First-Visit method to develop an optimal policy (π*), using a 5x5 matrix.

Part 2: Monte Carlo Algorithm
• Write a Python program that uses the MC Every-Visit method to develop an optimal policy (π*).

Part 3:
• Write a Python program that uses the QL Off-Policy method to develop an optimal policy (π*).

Part 4:
• Write a Python program that uses the QL On-Policy (SARSA) method to develop an optimal policy (π*).

Part 5:
• Write a Python program that uses the QL decaying epsilon-greedy method to develop an optimal policy (π*).

Part 6:
• Write a Python program that plots the Average Cumulative Reward curves for Parts 3, 4, and 5.
'''

import numpy as np
from matplotlib import pyplot as plt
import os
import random
from astropy.table import Table


#initialize empty nxn grid
def emptyGrid(gridSize):
    return np.zeros((gridSize,gridSize))
    

#create initial state
def create_initState(gridSize, terminationStates):
    
    rand_row = random.randint(0,gridSize-1)
    rand_col = random.randint(0,gridSize-1)
    
    #continue iterating until coordinates are found which are in neither termination states
    while [rand_row,rand_col] in terminationStates:
        
        rand_row = random.randint(0,gridSize-1)
        rand_col = random.randint(0,gridSize-1)
        
    init_state = [rand_row,rand_col]
    
    return init_state


#define dictionary that represent the possible directions
#possible integer directions (1 --> left, 2 --> right, 3 --> up, 4 --> down)
#correspond to either adding or subtracting 1 from a column (lateral) or row (vertical)
def direction_dict():
    
    num_array = [1,2,3,4]
    index_change = [-1,1,-1,1]
    direction_dictionary = dict(zip(num_array,index_change))
    
    return direction_dictionary


#now. which direction shall the agent random walk?
def nextState(current_state, gridSize):
    
    #"initialize" a next_state; important for the while loop
    next_state = current_state
    
    #define direction dictionary; gives 1 or -1 depending on direction integer 1-4 (see below)
    d = direction_dict()
        
    #might be the case that leftright = 0. if so, use while statement to prevent agent from moving diagonally
    #(i.e., both a lateral and vertical motion)
    while next_state == current_state:
        direction = random.choice([1,2,3,4])  #1 for left, 2 for right; 3 for up, 4 for down
        
        #if direction is a lateral motion, either 1 or 2 (not 3 or 4),
        if direction not in [3,4]:
            next_state = [current_state[0], current_state[1]+d[direction]] if ((current_state[1]+d[direction] < gridSize) & (current_state[1]+d[direction] >= 0)) else next_state
        
        #if direction is a vertical motion, either 3 or 4 (not 1 or 2),
        else:
            next_state = [current_state[0]+d[direction], current_state[1]] if ((current_state[0]+d[direction] < gridSize) & (current_state[0]+d[direction] >= 0)) else next_state
    
    return next_state


#oh, look. another while loop.
#so long as the next state is not in either of the termination states, continue random walk
#create "initial next_state" (important for initializing the while loop) as well as an initial current_state
def oneEpisodeWalk(gridSize, terminationStates):

    init_state = create_initState(gridSize, terminationStates)
    
    current_state = init_state

    #create list of every random walk for one episode; first item is, of course, the initial state.
    random_walks = [init_state]

    while current_state not in terminationStates:

        #generate next random walk
        next_state = nextState(current_state,gridSize)

        #append this next state to the random_walks list
        random_walks.append(next_state)

        #set current state as the next state
        current_state = next_state
        
    return(random_walks)


#I first require a list to track which spaces have already been traversed
#the value (length of this array + 1) will be the number I subtract from len(random_walks) to yield k_steps
#the + 1 is to prevent the program from including the termination state, since we don't want to include the
#termination state in the calculation of G (the rewardState = 0 here). 'Tis a product of how I wrote the function.
#This list is built into used_spaces if no other list is specified.
#returns all G(s) values for everyVisit (G_ALL) and unique G(s) for firstVisit (G_all)
def calc_G(gamma, rewardSize, random_walks, terminationStates, G_ALL=[], G_all=[], used_spaces=[]):
    
    #set up G counter
    G = 0
    
    #k_steps represents the number of steps between the current step and termination state
    k_steps = len(random_walks) - (len(used_spaces)+1)
        
    #if k_steps = 0, then we have arrived at the termination state. set G = 0.
    if k_steps == 0:
        G = 0
    
    #otherwise, loop through every step of the random walk, beginning at the current step space
    else:
        
        for n in range(k_steps):
            g = (gamma**n)*(rewardSize)
            G += g
            
    #will need an account of EVERY G FOR EVERY STEP.
    G_ALL.append(G)
    
    #include G in G_all list, but only if virst visit
    if random_walks[len(random_walks)-(k_steps+1)] not in used_spaces:
        G_all.append(G)
    
    used_spaces.append(random_walks[len(random_walks)-(k_steps+1)])

    #include the current space's coordinates in the used_spaces list
    
    return(G_ALL, G_all, used_spaces)


#continuously updates G_ALL, G_all, and used_spaces depending on how many indices are in random_walks
def populateGgrid(random_walks, rewardSize, gamma, terminationStates):
    
    for i in range(len(random_walks)):
        if i == 0:
            G_ALL, G_all, used_spaces = calc_G(gamma, rewardSize, random_walks, terminationStates, G_ALL=[], G_all=[], used_spaces=[])
        else:
            G_ALL, G_all, used_spaces = calc_G(gamma, rewardSize, random_walks, terminationStates, G_ALL, G_all, used_spaces)
    
    return G_ALL, G_all, used_spaces


#the following function finds 'unique' coordinates and, unlike numpy.unique, preserves order
def createFirstVisitList(random_walks):
    
    first_visits = []
    
    for coordinate in random_walks:
             if coordinate not in first_visits:
                 first_visits.append(coordinate)
    
    return first_visits


#update Ngrid!
def Ngrid_firstVisit(random_walks, Ngrid_current, terminationStates):
    
    #for firstVisit, only add +1 for the, well, first visit of the agent to that space
    
    #create unique list of coordinates which agent visited
    first_visits = createFirstVisitList(random_walks)
    #print(first_visits)
    
    #update each coordinate in first_visits
    for coord in first_visits:
        
        n = coord
        Ngrid_current[n[0],n[1]] += 1 if n not in terminationStates else 0
        
    return Ngrid_current
    
    
#update Sgrid!
def Sgrid_firstVisit(random_walks, Sgrid_current, G_all, terminationStates):
    
    #for first visit, only include the G value generated during the agent's...first visit...to that space.
    #note that in G_all (first visit), there is one G per UNIQUE space.
    
    #create unique list of coordinates which agent visited
    first_visits = createFirstVisitList(random_walks)
    
    #update each coordinate in first_visits
    for i in range(len(G_all)):
        
        n = first_visits[i]
        
        Sgrid_current[n[0],n[1]] += G_all[i] if n not in terminationStates else 0
    
    return Sgrid_current
    
    
def Vgrid_generator(Sgrid,Ngrid):
    
    Vgrid = Sgrid/Ngrid
    
    #np.nan_to_num converts the 0/0 nan elements to zeros
    return(np.nan_to_num(Vgrid))


#one option for a convergence test
def convergence_test(prev_Vgrid, current_Vgrid, gridSize):
    
    #isolate spaces for which both grids have nonzero elements
    nozeros = (prev_Vgrid != np.zeros((gridSize,gridSize))) & (current_Vgrid != np.zeros((gridSize,gridSize)))
    
    #calculate np.abs(mean) of the difference between corresponding elements in the two grids
    error = np.abs(np.mean(np.sum(np.round(prev_Vgrid[nozeros],2) - np.round(current_Vgrid[nozeros],2))))
        
    return error


#use a larger box and tie it with a more grandiose bow
def policyGridMC(number_episodes, gamma=0.90,
    rewardSize=-1, gridSize=5, error_param=0.05, first_or_every='first'):

    #error list for convergence testing purposes
    #ensure error array is same length as number of episodes
    err=np.zeros(number_episodes)
    
    # parameter
    terminationStates = [[0,0], [gridSize-1, gridSize-1]]

    #create empty grid, will use as initialization of Sgrid, Ngrid, and Vgrid.
    ref_grid = emptyGrid(gridSize)
    
    #print Episode 0 grids (requirement of the assignment)
    print('##### EPISODE 0 #####')
    print('N(s), S(s), V(s)')
    print(ref_grid)
    print()
    print()
    
    for episode in range(1,number_episodes+1):
        
        if episode > 1:
            old_Vgrid = Vgrid.copy()
        
        #single episode random walks
        random_walks = oneEpisodeWalk(gridSize, terminationStates)
        
        #create G(s) list for current episode, as well as list of coordinates visited AT LEAST ONCE
        G_ALL, G_all, used_spaces = populateGgrid(random_walks, rewardSize, gamma, terminationStates)

        #in the case episode>1, be sure to use the CURRENT GRIDS, rather than the empty grid
        
        if first_or_every == 'first':
            Ngrid = Ngrid_firstVisit(random_walks, ref_grid.copy(), terminationStates) if episode==1 else Ngrid_firstVisit(random_walks, Ngrid, terminationStates)
            Sgrid = Sgrid_firstVisit(random_walks, ref_grid.copy(), G_all, terminationStates) if episode==1 else Sgrid_firstVisit(random_walks, Sgrid, G_all, terminationStates)
            Vgrid = Vgrid_generator(Sgrid, Ngrid)
        
        if first_or_every == 'every':
            Ngrid = Ngrid_firstVisit(random_walks, ref_grid.copy(), terminationStates) if episode==1 else Ngrid_firstVisit(random_walks, Ngrid, terminationStates)
            Sgrid = Sgrid_firstVisit(random_walks, ref_grid.copy(), G_all, terminationStates) if episode==1 else Sgrid_firstVisit(random_walks, Sgrid, G_all, terminationStates)
            Vgrid = Vgrid_generator(Sgrid, Ngrid)
                
        #let final_episode be number_episodes, in case convergence condition in the 'if' statement below fails
        final_episode = number_episodes
        
        #if two or more episodes occurred, apply the convergence test to determine the relative difference between current episode's and previous episode's Vgrids.
        if episode > 1:
            error = convergence_test(old_Vgrid, Vgrid, gridSize)
            err[episode-1] = error
            
            #somewhat arbitrary here: if 20 episodes have already passed (meaning both grids are relatively well-populated) and both the current episode and previous episode's errors are both smaller than the user-given error parameter, then assume a relative convergence has been reached.
            if (episode>20) & (error<error_param) & (err[episode-2]<error_param):

                final_episode = episode
    
        #now. for the first, tenth, and final episodes, I am required to print a few items:
        #table showing k, s, r, γ, and G(s) for all values of k
        #Ngrid, Sgrid, Vgrid.
        #methinks I will generate a table using astropy.
        
        #will either print the final_episode cutoff if convergence condition is met, or the maximum iteration episode which the user specifies
        if episode in [1,10,final_episode]:

            k = np.arange(1,len(random_walks)+1,1)
            r = np.zeros(len(random_walks))+rewardSize
            s_rows = np.asarray(random_walks)[:,0]
            s_cols = np.asarray(random_walks)[:,1]
            #print(random_walks)
            gamma_tab = np.zeros(len(random_walks))+gamma
            Gs = G_ALL
            tab = Table([k,s_rows,s_cols,r,gamma_tab,Gs],names=['k','row','col','r','γ','G(s)'])
            
            print('##### EPISODE {} #####'.format(str(episode)))
            print('TABLE')
            print(tab)
            print()
            print('N(s) grid')
            print(Ngrid)
            print()
            print('S(s) grid')
            print(np.round(Sgrid,2))
            print()
            print('V(s) grid')
            print(np.round(Vgrid,2))
            print()
            print()
            
            if episode == final_episode:
                break

###################################################################

#I now must perform a similar task, but with the MC Every Visit approach. This technique consists of tracking EVERY TIME the agent visits a space per episode, rather than only the first visit. For instance, for some episode assume the space visits tile [3,3] four times. I then extract the *average* G(s) for these visits to [3,3] during this one episode: sum[G(s)] / [num_visits].

#for readability purposes, I will simply recreate the functions above that were previously tailored to the firstVisit technique.

#update Ngrid!

def Ngrid_everyVisit(random_walks, Ngrid_current, terminationStates):
    
    #for everyVisit, add +1 for every instance of the agent random-walking to that space in an episode
    
    #update each coordinate in first_visits
    for coord in random_walks:
        
        n = coord
        Ngrid_current[n[0],n[1]] += 1 if n not in terminationStates else 0
        
    return Ngrid_current


def Sgrid_everyVisit(random_walks, Sgrid_current, G_ALL, terminationStates):
    
    #include the G value generated during the agent's...EVERY visit...to that space.
    #G_ALL contains these values.
    
    #update each coordinate in Sgrid
    for i in range(len(G_ALL)):
        
        #isolate random_walks coordinate corresponding with the current G(s)
        n = random_walks[i]
        
        Sgrid_current[n[0],n[1]] += G_ALL[i] if n not in terminationStates else 0
    
    return Sgrid_current

###################################################################

#Q-Learning Zeit. I decide to use matrix indices 0-24, rather than coordinates.

#left --> n-1 if n-1 in row
#right --> n+1 if n+1 in row
#up --> n-gridSize if n-gridSize in column
#down --> n+gridSize if n+gridSize in column

def setDictionary(gridSize,terminationIndices):
    
    #create reference grid, where each space is some consecutive integer value from 0 to (gridSize+1)**2
    ref_index_grid = (np.arange(0,(gridSize)**2,1)).reshape((gridSize,gridSize))
    
    #create first dictionary entries
    indices = np.arange(0,(gridSize)**2,1)
    
    #create second dictionary entries
    movements = []
    
    #for every row, create array of indices to which agent can travel
    for row in range(gridSize):
                
        for col in range(gridSize):
            
            space_movements = []
            
            n = ref_index_grid[row,col]

            if (n-1>=0) & (n-1<=np.max(indices)) & (n-1 in ref_index_grid[row,:]):
                space_movements.append(n-1)
            if (n+1>=0) & (n+1<=np.max(indices)) & (n+1 in ref_index_grid[row,:]):
                space_movements.append(n+1)
            if (n-gridSize>=0) & (n-gridSize<=np.max(indices)):
                space_movements.append(n-gridSize)
            if (n+gridSize>=0) & (n+gridSize<=np.max(indices)):
                space_movements.append(n+gridSize)
            
            #terminal states can loop onto themselves
            if n in terminationIndices:
                space_movements.append(n)
            
            movements.append(space_movements)
    
    return dict(zip(indices,movements))


def setRewardMatrix(gridSize, movement_dictionary, terminationIndices):
    
    #create nxn blank canvas
    canvas = np.zeros((gridSize**2,gridSize**2))
    
    #create reference grid, where each space is some consecutive integer value from 0 to (gridSize)**2
    ref_index_grid = (np.arange(0,(gridSize)**2,1)).reshape((gridSize,gridSize))
    
    #I realize the following approach is not 'ideal', but the stress-induced extinguishing of creativity on my end
    #has me resorting to pixel-by-pixel loops.
    #goal: for every reward index coordinate, determine whether state row (n) can go to action (m). if yes,
    #[n,m] = 0. if no, [n,m] = -1.
    #if n can go to terminal state, [n,terminal_state] = 100.
    
    #for every state in index_grid (0-24),
    for state in range(len(canvas)):
        
        #extract the possible movements of this state (i.e., which spaces the agent can move to)
        list_of_moves = movement_dictionary[state]

        #for every action in this row (also 0-24),
        for action in range(len(canvas)):
            
            #determine whether the action is one of the possible agent movements. if yes, set = 0 unless
            #the action is in one of the terminal states
            
            if action in list_of_moves:
                canvas[state,action] = 0 if action not in terminationIndices else 100
                
            if action not in list_of_moves:
                canvas[state,action] = -1
    
    return canvas


#CRUDE EXAMPLE: say initial state is 6
#can go to 0,7,12
#randomly choose 7
#can go to 1,6,8,13

#Q[6,7] = R[6,7] + Gamma*Max(Q[7,1],Q[7,6],Q[7,8],Q[7,13])

def randomWalkOne(gridSize, Gamma, direction_dictionary, Qgrid, rewardMatrix, init_state=None):
    
    #if an initial state is given, as with the case of the continuation of a randomwalk, then apply user input
    if init_state is not None:
        init_state=init_state
    else:
        init_state = random.randint(0,gridSize**2-1)  #initial state
    
    init_list = direction_dictionary[init_state]  #list of possible next moves
    next_state = random.choice(init_list)         #next move
    next_list = direction_dictionary[next_state]  #list of possible next, next moves
    
    #create list of current Qgrid values according to next_list above; will be as many as there are possible moves
    #from next_state. these are needed for the np.max hoohaw.
    Qgrid_vals = np.zeros(len(next_list))
    for index in range(len(next_list)):
        Qgrid_vals[index] = Qgrid[next_state,next_list[index]]
    
    #update value of Q element
    Q_val = rewardMatrix[init_state,next_state] + Gamma*np.max(Qgrid_vals)

    #update Qgrid
    Qgrid[init_state,next_state] = Q_val
    
    #print('init state',init_state)
    #print('next state',next_state)
    
    #isolate reward; required for Part Six of Assignment 8
    reward = rewardMatrix[init_state,next_state]
    
    return Qgrid, next_state, reward


def oneEpisode(gridSize, Gamma, direction_dictionary, rewardMatrix, Q_init):
    
    #define termination states
    terminationStates = [[0,0], [gridSize**2-1,gridSize**2-1]]
    
    #set up init_state and next_state variables, in order for the while loop to work correctly
    #I am fatigued, so I shall settle on corny strings.
    init_state = 'hoo'
    next_state = 'haw'
    
    iteration_counter = 0
    reward_counter = 0
    
    #continue looping until goal state
    while [init_state,next_state] not in terminationStates:
        
        #if first time, then init_state must be randomly determined
        if iteration_counter == 0:
            Qgrid,next_state,reward = randomWalkOne(gridSize, Gamma, direction_dictionary, Q_init, rewardMatrix, init_state=None)
            
            iteration_counter += 1
            reward_counter += reward
            
        #if not first time, then init_state is simply the next_state
        else:
            init_state = next_state
            Qgrid,next_state,reward = randomWalkOne(gridSize, Gamma, direction_dictionary, Qgrid, rewardMatrix, init_state=init_state)
            
            iteration_counter += 1
            reward_counter += reward
            
        average_reward = reward_counter/iteration_counter
        
    return Qgrid, average_reward


def convergence_test_QL(prev_Qgrid, current_Qgrid, gridSize):
    
    #normalize grids
    prev_Qgrid = prev_Qgrid/np.max(prev_Qgrid)
    current_Qgrid = current_Qgrid/np.max(current_Qgrid)
    
    nozeros = (prev_Qgrid != np.zeros((gridSize**2,gridSize**2))) & (current_Qgrid != np.zeros((gridSize**2,gridSize**2)))
    
    #one idea for determining error...
    error = np.abs(np.mean(np.sum(np.round(prev_Qgrid[nozeros],2) - np.round(current_Qgrid[nozeros],2))))
        
    return error


#let's put it together! CONGLOMERATE.
def QLearningGridWorld(n_iterations=1000, gridSize=3, Gamma=0.9, error_param = 0.05, partsix=False):

    #prepare parameters
    gridSize=gridSize
    Gamma=Gamma
    terminationIndices=[0,gridSize**2-1]

    #set initial Qgrid
    Q_init = np.zeros((gridSize**2,gridSize**2))
    
    print('Episode Zero Qgrid')
    print(Q_init)
    print()
    
    Qgrid_update = Q_init.copy()
    
    #ensure error list is same length as n_iterations
    err = np.zeros(n_iterations)

    #set dictionary
    movement_dictionary = setDictionary(gridSize,terminationIndices)

    #set reward matrix
    rewardMatrix = setRewardMatrix(gridSize, movement_dictionary, terminationIndices)
    
    print('Reward Matrix')
    print(rewardMatrix)
    print()
    
    #Part Six Hoohaw
    cumulative_average_reward = []
    total = 0
    
    #loop over every n_iteration, at least until convergence condition is satisfied
    for episode in range(n_iterations):
        
        episode=episode+1
        
        #define previous Qgrid, only if episode != 0
        if episode > 1:
            prev_Qgrid = Qgrid_update.copy()
        
        #create new episode
        Qgrid_update += oneEpisode(gridSize, Gamma, movement_dictionary, rewardMatrix, Q_init)[0]
            
        average_reward = oneEpisode(gridSize, Gamma, movement_dictionary, rewardMatrix, Q_init)[1]
        
        total+=average_reward
        cumulative_average_reward.append((total-average_reward)/episode)
        
        #let final_episode be maximum possible iteration, in case convergence not reached in 'if' statement
        final_episode = n_iterations
        
        
        if episode > 10:
            error = convergence_test_QL(prev_Qgrid, Qgrid_update, gridSize)
            err[episode-1] = error
            
            if ((Qgrid_update[0,0]/np.max(Qgrid_update)*100 >= 99) & (Qgrid_update[gridSize**2-1,gridSize**2-1]/np.max(Qgrid_update)*100 >= 99)) & (error<error_param):

                final_episode = episode
                
        if episode in [1,10,final_episode]:
            
            print('##### EPISODE {} #####'.format(str(episode)))
            
            print('QGrid (Normalized)')
            print(np.round(np.round(Qgrid_update,1)/np.max(np.round(Qgrid_update,1))*100,2))
            print()
            
            if episode == final_episode:
                break
    
    if partsix==True:
        return cumulative_average_reward
###################################################################

#SARSA...same idea as Q-learning above, but now next states are taken with the maximum estimated value (on-policy)
#*Could* I combine the QL and SARSA functions to reduce the size of this program and thus improve readability? Yes. Will I do so? No.

def randomWalkOne_SARSA(gridSize, Gamma, direction_dictionary, Qgrid, rewardMatrix, init_state=None):
    
    #if an initial state is given, as with the case of the continuation of a randomwalk, then apply user input
    if init_state is not None:
        init_state=init_state
    else:
        init_state = random.randint(0,gridSize**2-1)  #initial state
    
    init_list = direction_dictionary[init_state]  #list of possible next moves
    
    #for every possible next move, find the corresponding Qgrid value and compare with the maxQ
    #set this value as the next state, if applicable
    
    #create log of every Qvalue
    Qvalue_all = []
        
    for i in range(len(init_list)):

        Qvalue_initlist = Qgrid[init_state,init_list[i]]
        Qvalue_all.append(Qvalue_initlist)

        if i == 0:
            maxQ = 0

        #if the current Qvalue is larger than the maxQ, set next action to init_list
        if Qvalue_initlist > maxQ:
            next_state = init_list[i]
            maxQ = Qvalue_initlist

    #if there are duplicates somewhere...tend to it.
    #note: next_state will only change if the conditions are met.
    if (len(np.unique(Qvalue_all)) != len(init_list)):

        #identify all unique Qvalues and the number of times they occur
        unique, counts = np.unique(Qvalue_all, return_counts=True)
        
        #the 'duplicate Q value' is that which is the maximum Q value of those that appear most often
        dup_Q = np.max(unique[counts==np.max(counts)])

        #if the duplicate Q value is either larger than or equal to the current maxQ,
        #randomly choose from the corresponding actions and set the chosen as the next state
        if dup_Q >= maxQ:
            dup_action = np.asarray(init_list)[(np.asarray(Qvalue_all) == dup_Q)]
            next_state = random.choice(dup_action)
    
    #next_state = random.choice(init_list)         #next move
    next_list = direction_dictionary[next_state]  #list of possible next, next moves
    
    #create list of current Qgrid values according to next_list above; will be as many as there are possible moves
    #from next_state. these are needed for the np.max hoohaw.
    Qgrid_vals = np.zeros(len(next_list))
    for index in range(len(next_list)):
        Qgrid_vals[index] = Qgrid[next_state,next_list[index]]
    
    #update value of Q element
    Q_val = rewardMatrix[init_state,next_state] + Gamma*np.max(Qgrid_vals)

    #update Qgrid
    Qgrid[init_state,next_state] = Q_val
    
    #isolate reward; required for Part Six of Assignment 8
    reward = rewardMatrix[init_state,next_state]
    
    return Qgrid, next_state, reward


def oneEpisode_SARSA(gridSize, Gamma, direction_dictionary, rewardMatrix, Q_init):
    
    #define termination states
    terminationIndices = [0, gridSize**2-1]
    
    #set up init_state and next_state variables, in order for the while loop to work correctly
    #I am fatigued, so I shall settle on corny strings.
    init_state = 'hoo'
    next_state = 'haw'
    
    iteration_counter = 0
    reward_counter = 0
    
    #continue looping until goal state
    while next_state not in terminationIndices:
        
        #if first time, then init_state must be randomly determined
        if iteration_counter == 0:
            Qgrid,next_state,reward = randomWalkOne_SARSA(gridSize, Gamma, direction_dictionary, Q_init, rewardMatrix, init_state=None)
            
            iteration_counter += 1
            reward_counter += reward
        
        #if not first time, then init_state is simply the next_state
        else:
            init_state = next_state
            Qgrid,next_state,reward = randomWalkOne_SARSA(gridSize, Gamma, direction_dictionary, Qgrid, rewardMatrix, init_state=init_state)
            
            iteration_counter += 1
            reward_counter += reward
    
        average_reward = reward_counter/iteration_counter
        
    return Qgrid, average_reward


#let's put it together! CONGLOMERATE.
def QLearningGridWorld_SARSA(n_iterations=1000, gridSize=3, Gamma=0.9, error_param = 0.05, partsix=False):

    #prepare parameters
    gridSize=gridSize
    Gamma=Gamma
    terminationIndices=[0,gridSize**2-1]

    #set initial Qgrid
    Q_init = np.zeros((gridSize**2,gridSize**2))
    
    print('Episode Zero Qgrid')
    print(Q_init)
    print()
    
    Qgrid_update = Q_init.copy()
    
    #ensure error list is same length as n_iterations
    err = np.zeros(n_iterations)

    #set dictionary
    movement_dictionary = setDictionary(gridSize,terminationIndices)

    #set reward matrix
    rewardMatrix = setRewardMatrix(gridSize,movement_dictionary,terminationIndices)
    
    print('Reward Matrix')
    print(rewardMatrix)
    print()
    
    #Part Six Hoohaw
    cumulative_average_reward = []
    total=0
    
    #loop over every n_iteration, at least until convergence condition is satisfied
    for episode in range(n_iterations):
        
        episode=episode+1
        
        #define previous Qgrid, only if episode != 0
        if episode > 1:
            prev_Qgrid = Qgrid_update.copy()
        
        #create new episode
        Qgrid_update += oneEpisode_SARSA(gridSize, Gamma, movement_dictionary, rewardMatrix, Q_init)[0]
        
        average_reward = oneEpisode_SARSA(gridSize, Gamma, movement_dictionary, rewardMatrix, Q_init)[1]
            
        total+=average_reward
        cumulative_average_reward.append((total - average_reward)/episode)
        
        #let final_episode be maximum possible iteration, in case convergence not reached in 'if' statement
        final_episode = n_iterations
        
        
        if episode > 50:
            error = convergence_test_QL(prev_Qgrid, Qgrid_update, gridSize)
            err[episode-1] = error
            
            if (error<error_param):

                final_episode = episode
                
        if episode in [1,10,final_episode]:
            
            print('##### EPISODE {} #####'.format(str(episode)))
            
            print('QGrid')
            print(np.round(np.round(Qgrid_update,1)/np.max(np.round(Qgrid_update,1))*100,2))
            print()
            
            if episode == final_episode:
                break
    
    if partsix == True:
        return cumulative_average_reward
 
###################################################################

#Epsilon-Greedy...same idea as Q-learning above, but now next states are taken with the maximum estimated value (on-policy) OR randomly (off-policy), depending on the epsilon value.
#*Could* I combine the QL and SARSA functions to reduce the size of this program and thus improve readability? Yes. Will I do so? No.

def calc_epsilon(episode,c=100):
    return np.exp(-(episode-1)/c)

def randomWalkOne_greedy(gridSize, Gamma, epsilon, direction_dictionary, Qgrid, rewardMatrix, init_state=None):
    
    #if an initial state is given, as with the case of the continuation of a randomwalk, then apply user input
    if init_state is not None:
        init_state=init_state
    else:
        init_state = random.randint(0,gridSize**2-1)  #initial state
    
    init_list = direction_dictionary[init_state]  #list of possible next moves
    
    #for every possible next move, find the corresponding Qgrid value and compare with the maxQ
    #set this value as the next state, if applicable
    
    #generate random number 0<n<1
    rand_n = random.uniform(0, 1)
    
    #if less than epsilon, then randomly select next state from the list of possible actions
    if rand_n < epsilon:
        next_state = random.choice(init_list)
    
    #if >= epsilon, then select next action based on highest Qval
    else:
        #create log of every Qvalue
        Qvalue_all = []

        #for every possible action...
        for i in range(len(init_list)):

            Qvalue_initlist = Qgrid[init_state,init_list[i]]
            Qvalue_all.append(Qvalue_initlist)

            #if first iteration, then initialize a maximum Q value
            if i == 0:
                maxQ = 0

            #if the current Qval is larger than the maxQ, set next action to init_list and update maxQ
            if Qvalue_initlist > maxQ:
                next_state = init_list[i]
                maxQ = Qvalue_initlist

        #if there are duplicates somewhere...tend to it.
        #note: next_state should only change if the conditions are met.

        #if the number of unique Qvals is not equal to the number of all Qvals
        if (len(np.unique(Qvalue_all)) != len(init_list)):

            #identify all unique Qvalues and the number of times they occur
            unique, counts = np.unique(Qvalue_all, return_counts=True)

            #the 'duplicate Q value' (dup_Q) is the maximum Q value of those that appear most often

            dup_Q = np.max(unique[counts==np.max(counts)])

            #if the duplicate Q value is equal to the current maxQ,
            #randomly choose from the corresponding actions and set the chosen as the next state
            if dup_Q >= maxQ:

                dup_action = np.asarray(init_list)[(np.asarray(Qvalue_all) == dup_Q)]
                next_state = random.choice(dup_action)
    
    next_list = direction_dictionary[next_state]  #list of possible next, next moves
    
    #create list of current Qgrid values according to next_list above; will be as many as there are possible moves
    #from next_state. these are needed for the np.max hoohaw.
    Qgrid_vals = np.zeros(len(next_list))
    for index in range(len(next_list)):
        Qgrid_vals[index] = Qgrid[next_state,next_list[index]]
    
    #update value of Q element
    Q_val = rewardMatrix[init_state,next_state] + Gamma*np.max(Qgrid_vals)

    #update Qgrid
    Qgrid[init_state,next_state] = Q_val

    #isolate reward; required for Part Six of Assignment 8
    reward = rewardMatrix[init_state,next_state]

    return Qgrid, next_state, reward


def oneEpisode_greedy(gridSize, Gamma, direction_dictionary, rewardMatrix, Q_init, epsilon):
    
    #define termination states
    terminationIndices = [0, gridSize**2-1]
    
    #set up init_state and next_state variables, in order for the while loop to work correctly
    #I am fatigued, so I shall settle on corny strings.
    init_state = 'hoo'
    next_state = 'haw'
    
    iteration_counter = 0
    reward_counter = 0
    
    #continue looping until goal state
    while next_state not in terminationIndices:
        
        #if first time, then init_state must be randomly determined
        if iteration_counter == 0:
            Qgrid,next_state,reward = randomWalkOne_greedy(gridSize, Gamma, epsilon, direction_dictionary, Q_init, rewardMatrix, init_state=None)

            iteration_counter += 1
            reward_counter += reward
            
        #if not first time, then init_state is simply the next_state
        else:
            
            init_state = next_state
            Qgrid,next_state,reward = randomWalkOne_greedy(gridSize, Gamma, epsilon, direction_dictionary, Qgrid, rewardMatrix, init_state=init_state)
            
            iteration_counter += 1
            reward_counter += reward
            
    average_reward = reward_counter/iteration_counter
            
            
    return Qgrid, average_reward
    
    
#let's put it together! CONGLOMERATE.
def QLearningGridWorld_greedy(n_iterations=1000, gridSize=3, Gamma=0.9, error_param = 0.05, partsix=False):

    #prepare parameters
    gridSize=gridSize
    Gamma=Gamma
    terminationIndices=[0,gridSize**2-1]

    #set initial Qgrid
    Q_init = np.zeros((gridSize**2,gridSize**2))
    
    print('Episode Zero Qgrid')
    print(Q_init)
    print()
    
    Qgrid_update = Q_init.copy()
    
    #ensure error list is same length as n_iterations
    err = np.zeros(n_iterations)

    #set dictionary
    movement_dictionary = setDictionary(gridSize,terminationIndices)

    #set reward matrix
    rewardMatrix = setRewardMatrix(gridSize,movement_dictionary,terminationIndices)
    
    print('Reward Matrix')
    print(rewardMatrix)
    print()
    
    #Part Six hoohaw.
    cumulative_average_reward = []
    total=0
    
    #loop over every n_iteration, at least until convergence condition is satisfied
    for episode in range(n_iterations):
        
        episode=episode+1
        
        epsilon= calc_epsilon(episode,c=500)
        
        #define previous Qgrid, only if episode != 0
        if episode > 1:
            prev_Qgrid = Qgrid_update.copy()
        
        #create new episode
        Qgrid_update += oneEpisode_greedy(gridSize, Gamma, movement_dictionary, rewardMatrix, Q_init, epsilon)[0]
        
        average_reward = oneEpisode_greedy(gridSize, Gamma, movement_dictionary, rewardMatrix, Q_init, epsilon)[1]
        
        total+=average_reward
        cumulative_average_reward.append((total - average_reward)/episode)
        
        #let final_episode be maximum possible iteration, in case convergence not reached in 'if' statement
        final_episode = n_iterations
        
        if episode > 2:
            error = convergence_test_QL(prev_Qgrid, Qgrid_update, gridSize)
            err[episode-1] = error
            
            if ((Qgrid_update[0,0]/np.max(Qgrid_update)*100 >= 95) & (Qgrid_update[gridSize**2-1,gridSize**2-1]/np.max(Qgrid_update)*100 >= 95)) & (error<error_param):

                final_episode = episode
                
        if episode in [1,10,final_episode]:
            
            print('##### EPISODE {} #####'.format(str(episode)))
            
            print('QGrid')
            print(np.round(np.round(Qgrid_update,1)/np.max(np.round(Qgrid_update,1))*100,2))
            print()
            
            if episode == final_episode:
                break
    
    if partsix==True:
        return cumulative_average_reward
    
####################################################
#Part Six...create average cumulative reward plot. I guess.
#Similar deal --> I could combine, but I choose not to.
#is the average cumulative reward equation correct? lawl idk. at least the plots are similar in shape to those in the lecture slides... y-axis values notwithstanding.

def gather_rewards(n_iterations=3000,gridSize=5):
    
    #set error_param=0 to ensure that all 3000 iterations are run.
    reward_part3 = QLearningGridWorld(n_iterations, gridSize, Gamma=0.9, error_param=0, partsix=True)
    reward_part4 = QLearningGridWorld_SARSA(n_iterations, gridSize, Gamma=0.9, error_param=0, partsix=True)
    reward_part5 = QLearningGridWorld_greedy(n_iterations, gridSize, Gamma=0.9, error_param=0, partsix=True)
    
    return reward_part3, reward_part4, reward_part5

def create_plot(homedir,dpi=300,n_iterations=3000):
    
    reward_part3, reward_part4, reward_part5 = gather_rewards(n_iterations=n_iterations)
    x = np.arange(0,n_iterations,1)
    
    plt.figure(figsize=(7,5))
    plt.plot(x,reward_part3,color='red',label='Off-Policy')
    plt.plot(x,reward_part4,color='green',label='SARSA')
    plt.plot(x,reward_part5,color='blue',label='decay epsilon-greedy')
    plt.legend(fontsize=15)
    plt.xlabel('Episode (t)',fontsize=20)
    plt.ylabel('Average Cumulative Reward',fontsize=20)
    plt.savefig(homedir+'/Desktop/partsix.png',dpi=dpi)
    plt.close()

if __name__ == '__main__':

    homedir = os.getenv("HOME")

    print('######## PART ONE ######## ')
    print()
    policyGridMC(number_episodes=5000, gamma=0.90, rewardSize=-1, gridSize=5, error_param=0.01)
    
    print('######## PART TWO ######## ')
    print()
    policyGridMC(number_episodes=5000, gamma=0.90, rewardSize=-1, gridSize=5, error_param=0.01, first_or_every='every')
    
    print('######## PART THREE ########')
    print()
    QLearningGridWorld(n_iterations=1000, gridSize=5, Gamma=0.9, error_param=0.01)

    print('######## PART FOUR ########')
    print()
    QLearningGridWorld_SARSA(n_iterations=1000, gridSize=5, Gamma=0.9, error_param=0.01)
    
    print('######## PART FIVE ########')
    print()
    QLearningGridWorld_greedy(n_iterations=5000, gridSize=5, Gamma=0.9, error_param=0.01)

    print('######## PART SIX ########')
    print()
    create_plot(homedir,dpi=300,n_iterations=5000)
    print('See partsix.png')
