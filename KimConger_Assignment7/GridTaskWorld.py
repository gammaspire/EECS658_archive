'''
EECS-658 Assignment 7

Part 1: RL Policy Iteration Algorithm
• Write a Python program that uses the RL Policy Iteration algorithm to develop an optimal policy (π*), using a 5x5 matrix.

Part 2: RL Value Iteration Algorithm
• Write a Python program that uses the RL Value Iteration algorithm to develop anoptimal policy (π*).
'''

import numpy as np
from matplotlib import pyplot as plt
import os


#routine to compare iteration k, k-1 policy matrices
def compare_matrices_policy(old_policy_matrix, new_policy_matrix):
    
        #select a few 'comparison indices'
        difference_matrix = np.abs(new_policy_matrix - old_policy_matrix)
        difference_row = difference_matrix[0]
        evaluation = [difference_row[1],difference_row[2],
                      difference_row[3],difference_row[4]]
        
        return(evaluation)


#perform RL policy iteration for 5x5 matrix to generate optimal 'policy path'
def gridtask_world_policy(k, grid_length, grid_width, suppress_print=True, gamma=1, prob=0.25, reward_transition=-1, reward_terminal=0):

    k_range=np.arange(1,k+1,1)

    #base policy matrix
    policy_matrix = np.zeros((grid_length,grid_width))
    
    print('~~~~~~~~~~~ iteration 0 ~~~~~~~~~~~')
    print(policy_matrix)
    print()
    print()
    
    #prepare the old, new policy matrix canvases
    policy_matrix_old = policy_matrix.copy()
    policy_matrix_new = policy_matrix.copy()

    #for k iterations, loop through every non-terminal matrix element

    #per instructions, only print first, tenth, and nth iteration policy grid
    #nth iteration will be where policy grid approximately converges (my discretion)...the condition I give is the first instance where percent_difference = 0. See final lines of this function.
    
    print_indices = [1,10]
    

    #for iterations 1 to k:
    for n in k_range:

        policy_matrix_old = policy_matrix_new.copy()

        for row in range(len(policy_matrix_old)):

            for element in range(len(policy_matrix_old[row])):

                #if the current element is a terminal state, then ensure the value remains zero
                if ((row==0) & (element==0)) | ((row==grid_length-1) & (element==grid_width-1)):

                    policy_matrix_new[row][element] = reward_terminal

                else:

                    #example: UP --> take value of element above current element, but only if that element exists;
                    #if not, robot strikes a wall and therefore assumes the 'up' value is its current value

                    up = policy_matrix_old[row-1][element] if (row-1) >= 0 else policy_matrix_old[row][element]
                    down = policy_matrix_old[row+1][element] if (row+1) <= 4 else policy_matrix_old[row][element]
                    left = policy_matrix_old[row][element-1] if (element-1) >= 0 else policy_matrix_old[row][element]
                    right = policy_matrix_old[row][element+1] if (element+1) <= 4 else policy_matrix_old[row][element]

                    #assign new value to the current element
                    policy_matrix_new[row][element] = reward_transition + 0.25*gamma*(up+left+right+down)
        
        #if first iteration, then set the evaluation_dat array of the three grid elements as the output
        if n == 1:
            evaluation_dat = compare_matrices_policy(policy_matrix_old, policy_matrix_new)
            evaluation_dat_all = evaluation_dat.copy()

        #else...append the evaluation_dat array to the existing matrix
        else:
            evaluation_dat_new = compare_matrices_policy(policy_matrix_old, policy_matrix_new)
            evaluation_dat_all = np.vstack((evaluation_dat_all, evaluation_dat_new))
    
        if not suppress_print:
            if n in print_indices:
                print('~~~~~~~~~~~ iteration',n,'~~~~~~~~~~~')
                print('Unrounded (used for calculations):')
                print(policy_matrix_new)
                print()
                print('Rounded:')
                print(np.round(policy_matrix_new,0))
                print()
                print()
    
        #if approx. convergence condition is met, then break loop and print final policy matrix
        if (n-1)>10:
            if (evaluation_dat_all[n-1][1] == 0):
                print('~~~~~~~~~~~ iteration',n,'~~~~~~~~~~~')
                print('Unrounded:')
                print(policy_matrix_new)
                print()
                print('Rounded:')
                print(np.round(policy_matrix_new,0))
                print()
                print()
                n_final = n
                break
    
    return(np.arange(0,n_final,1),evaluation_dat_all)


def compare_matrices_value(old_policy_matrix, new_policy_matrix, grid_length):
    
        #select a few 'comparison indices'
        difference_matrix = np.abs(new_policy_matrix - old_policy_matrix)
        
        #isolate first row
        difference_row = difference_matrix[0]
        
        #evaluate difference between elements with positions farthest from terminal states
        evaluation = difference_row[grid_length - 1]
        
        return(evaluation)


def gridtask_world_value(k, grid_length, grid_width, suppress_print=True, gamma=1, reward_transition=-1, reward_terminal=0):

    k_range=np.arange(1,k+1,1)

    #base policy matrix
    policy_matrix = np.zeros((grid_length,grid_width))
    
    print('~~~~~~~~~~~ iteration 0 ~~~~~~~~~~~')
    print(policy_matrix)
    print()
    print()
    
    policy_matrix_old = policy_matrix.copy()
    policy_matrix_new = policy_matrix.copy()

    #for k iterations, loop through every non-terminal matrix element

    #for iterations 1 to k:
    for n in k_range:

        policy_matrix_old = policy_matrix_new.copy()

        for row in range(len(policy_matrix_old)):

            for element in range(len(policy_matrix_old[row])):

                #if the current element is a terminal state, then ensure the value remains zero
                if ((row==0) & (element==0)) | ((row==grid_length-1) & (element==grid_width-1)):

                    policy_matrix_new[row][element] = reward_terminal

                else:

                    #example: UP --> take value of element above current element, but only if that element exists;
                    #if not, robot strikes a wall and therefore assumes the 'up' value is its current value

                    up = policy_matrix_old[row-1][element] if (row-1) >= 0 else policy_matrix_old[row][element]
                    down = policy_matrix_old[row+1][element] if (row+1) <= (grid_width-1) else policy_matrix_old[row][element]
                    left = policy_matrix_old[row][element-1] if (element-1) >= 0 else policy_matrix_old[row][element]
                    right = policy_matrix_old[row][element+1] if (element+1) <= (grid_length-1) else policy_matrix_old[row][element]

                    #assign new value to the current element
                    policy_matrix_new[row][element] = np.max([reward_transition+gamma*up, reward_transition+gamma*left, reward_transition+gamma*right, reward_transition+gamma*down])

        if n == 1:
            evaluation_dat = compare_matrices_value(policy_matrix_old, policy_matrix_new, grid_length)
            evaluation_dat_all = evaluation_dat.copy()

        else:
            evaluation_dat_new = compare_matrices_value(policy_matrix_old, policy_matrix_new, grid_length)
            evaluation_dat_all = np.vstack((evaluation_dat_all, evaluation_dat_new))
    
        print_indices = [1,2]

        if not suppress_print:
            if n in print_indices:
                print('~~~~~~~~~~~ iteration',n,'~~~~~~~~~~~')
                print(policy_matrix_new)
                print()
                print()
        
        if (n-1)>2:
            if (evaluation_dat_all[n-1][0] == 0):
                print('~~~~~~~~~~~ iteration',n,'~~~~~~~~~~~')
                print(policy_matrix_new)
                print()
                print()
                n_final = n
                break

    return(np.arange(0,n_final,1),evaluation_dat_all)


#compiles a figure of percent differences for three policy grid elements, as well as a marker for the k-value at which actual convergence occurs (minimum k where percent difference = 0).
def eval_policy_plot(homedir,k_range,evaluation_dat_all):
    plt.figure(figsize=(7,5))
    plt.plot(k_range,eval_dat_all[:,1],color='blue',label='Element (0,1)')
    plt.plot(k_range,eval_dat_all[:,2],color='red',label='Element (0,2)')
    plt.plot(k_range,eval_dat_all[:,3],color='green',label='Element (0,3)')
    
    try:
        conv_index = np.where(eval_dat_all[:,1] == 0)[0]
        conv_index = conv_index[0]
        plt.scatter(k_range[conv_index], (eval_dat_all[:,1])[conv_index], color='black', s=60, zorder=2, label='approx. convergence' +str(k_range[conv_index]))

    except:
        print('not enough iterations to find a 0% percent difference convergence point.')
    
    plt.xlabel('k iteration',fontsize=15)
    plt.ylabel(r'Element$_{k}$ - Element$_{k-1}$',fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(homedir+'/Desktop/KimConger_Assignment7/eval_policy.png')
    plt.close()


if __name__ == '__main__':

    homedir = os.getenv("HOME")
    
    print('PART ONE')
    print()
    k_range, eval_dat_all = gridtask_world_policy(1200, 5, 5, suppress_print=False)

    eval_policy_plot(homedir,k_range,eval_dat_all)

    print()
    
    print('PART TWO')
    print()
    k_range, eval_dat_all = gridtask_world_value(20, 5, 5,suppress_print=False)
