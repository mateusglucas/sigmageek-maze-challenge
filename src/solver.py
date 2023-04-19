#!/usr/bin/env python3

#
# Script to solve challenges 1, 2, 3 (trivial solution) and 4
#
# Usage:
#
# python3 solver.py N 
#
# where N is the desired challenge number (1-4)
#
# Elapsed times (after loading maze states):
#
# challenge 1: 33.61 s
# challenge 2: 57.01 s (forward propagation) / 206.37 s (forward + backward propagation) 
# challenge 3: 34.65 s 
# challenge 4: 33.72 s (until epoch 3999)
#

import sys
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from time import time
from scipy import sparse
import os

from position import Position
from automata_maze import AutomataMaze

plot_flag = False
plot_folder = '../images'
iters_per_info = 50 # info print interval, in iterations
iters_per_plot = 100

# Challenge 4 variables
unknown_block_solution_filename = 'challenge4_block_solution.txt'
unknown_block_start_row = 2300
unknown_block_start_col = 2300
   
UP = Position(-1,0)
DOWN = Position(1,0)
LEFT = Position(0,-1)
RIGHT = Position(0,1)

def plot(data, index, fig = 1):
    plt.figure(fig)
    plt.clf()
    plt.imshow(data, vmin = 0, vmax = 3 if n_lives>1 else 2)
    plt.savefig(plot_folder+'/'+image_prefix+str(index))
    plt.pause(0.001)    
    
def directions():
    return [UP, DOWN, LEFT, RIGHT]

maze_size = None
maze_shape = None
start_pos = None
end_pos = None
n_cols = None
 
def get_packed_maze_states(input_filename, states_filename, max_n_epochs):
    global maze_size
    global maze_shape
    global start_pos
    global end_pos
    global n_cols
    
    m = AutomataMaze(input_filename)
    if exists_unknown_block:
        solved_block = AutomataMaze(unknown_block_solution_filename)
        m.modify_maze(solved_block.maze, unknown_block_start_row, unknown_block_start_col) 
    
    maze_size = m.maze.size
    maze_shape = m.maze.shape
    start_pos = m.start_position()
    end_pos = m.end_position()
    n_cols = maze_shape[1]
    
    sz = maze_size//8 + (maze_size % 8 != 0)
    
    if os.path.isfile(states_filename):
        print('Found maze epochs file')
        print('Verifying dimensions...')
        packed_states = np.load(states_filename)
        if packed_states.shape == (max_n_epochs, sz):
            print('Dimensions are valid.')
            return packed_states
        print('Dimensions invalid. Creating new maze epochs file...')
        
    print('Creating maze epochs file...')
    packed_states = np.zeros((max_n_epochs, sz), dtype = np.uint8) 
    for i in range(max_n_epochs):
        if i%10 == 0:
            print('Saving epoch {}...'.format(i)+' '*10,end='\r')
        packed_states[i] = np.packbits(m.maze)
        m.next()
    print('Saving maze epochs file...')
    np.save(states_filename, packed_states)
    print('Maze epochs file saved.')
    return packed_states 

def is_valid(position):
    return position.row>=0 and position.row<maze_shape[0] and position.col>=0 and position.col<maze_shape[1]

def unpack(packed_values):
    return np.unpackbits(packed_values)[:maze_size].reshape(maze_shape) == 1

# To avoid using unpackbits when we only need one specific value
def get_bit(packed_values, row, col):
    elem_idx = col + row*n_cols
    byte_idx = elem_idx // 8
    bit_idx = 7 - (elem_idx % 8)       

    return (packed_values[byte_idx] & (1<<bit_idx)) != 0 
                  
def find_shortest_solution(max_n_epochs):
    start_time = time()
    
    # Load maze states
    print('Loading states...')
    start_load_time = start_time
    packed_states = get_packed_maze_states(input_filename, states_filename, max_n_epochs)
    end_load_time = time()
    print('Loading complete. Elapsed time: {:.2f}'. format(end_load_time-start_load_time))
    
    # Search for solution
    print('Searching solution...')
    start_time_search = time()
    
    packed_sz = maze_size//8 + (maze_size % 8 != 0)
    packed_positions = np.zeros((max_n_epochs, packed_sz), dtype = np.uint8)
    
    positions = np.full(maze_shape, False)
    if n_lives==1:
        last_positions = np.full((maze_shape[0]+2, maze_shape[1]+2), False)
    else:
        lives = np.zeros(maze_shape, dtype=np.int8)
        last_lives = np.zeros((maze_shape[0]+2, maze_shape[1]+2), dtype=np.int8)
       
    solution_found = False
    
    fig_idx = 0
    
    for i in range(max_n_epochs):
        last_epoch = i
        
        if (i % iters_per_info == 0):
            print('Epoch {}'.format(i))
        states = unpack(packed_states[i])

        if i==0:
            positions[start_pos.row, start_pos.col] = True
            if n_lives>1:
                lives[start_pos.row, start_pos.col] = n_lives
        else:
            if n_lives==1:
                positions = last_positions[1:-1,:-2] | last_positions[1:-1,2:] | last_positions[:-2,1:-1] | last_positions[2:,1:-1]
                positions = positions & (~states)
            else:
                lives = np.maximum.reduce([last_lives[1:-1,:-2], last_lives[1:-1,2:] , last_lives[:-2,1:-1], last_lives[2:,1:-1]])
                lives = lives - states
                lives = lives * (lives>0)
                positions = lives>0
                
        packed_positions[i] = np.packbits(positions)
        if n_lives==1:
            last_positions[1:-1,1:-1] = positions
        else:
            last_lives[1:-1,1:-1] = lives
        
        # No particles left
        if positions.any()==False:
            if plot_flag == True:
                plot(states+2*(positions+0), fig_idx, 1)
                fig_idx+=1
            print('No particles left!')
            break
        
        if plot_flag == True and i % iters_per_plot == 0:
            plot(states+2*(positions+0), fig_idx, 1)
            fig_idx+=1
        
        # Solution found
        if positions[end_pos.row, end_pos.col] == True:
            if plot_flag == True:
                plot(states+2*(positions+0), fig_idx, 1)
                fig_idx+=1
                
            if n_lives>1:
                print('Remaining lives: {}'.format(lives[end_pos.row, end_pos.col]))
            end_time_search = time()
            print('Solution found! (iter {}, total time: {:.2f} s)'.format(i, end_time_search-start_time_search))
            solution_found = True
            break   
   
    if solution_found == True:
        solution_idx = last_epoch
        last_position = end_pos
    else:
        # Find closest solution
        end_time_search = time()
        print('Solution not found. (iter {}, total time: {:.2f} s)'.format(i, end_time_search-start_time_search))
        print('Calculating closest solution...')
        
        # Matrix to store the first epoch that a position has been occcupied
        # Storing epoch + 1, to represent "no epoch" as 0 (positions 
        # that were never occupied), epoch 0 as 1, n-th epoch as n+1
        positions = unpack(packed_positions[last_epoch])
        positions_first_epoch = positions * (last_epoch+1)
        for i in range(last_epoch-1,-1,-1):
            positions = unpack(packed_positions[i])
            positions_first_epoch= positions_first_epoch*(~positions) + positions*(i+1)
        
        # Matrix with the distance from each position that has been 
        # occupied to the end position. Positions that were never
        # occupied will have distance 0
        distances = np.zeros(maze_shape)
        for row in range(maze_shape[0]):
            for col in range(maze_shape[1]):
                if positions_first_epoch[row,col]!=0:
                    distances[row,col] = (Position(row,col)-end_pos).norm()
        # Make distance of positions that were never occupied equal to
        # max distance + 1, to be able to find the minimum distance of 
        # the occupied positions
        distances = distances + (distances==0)*(distances.max()+1)
         
        # Find position closest to the end position and the first epoch 
        # that this position has been occupied
        min_index = np.unravel_index(distances.argmin(), distances.shape)
        solution_idx = positions_first_epoch[min_index]-1 # subtract 1 because the stored value is epoch + 1
        print('Closest distance {} on epoch {}'.format(distances.min(), solution_idx))   
        last_position = Position(*min_index)
    
    if n_lives==1:
        # If lives==1, walk backwards to find the path
        print('Calculating solution...')
        solution = ''
        for i in range(solution_idx-1,-1,-1):
            for d in directions():
                previous_position = last_position - d
                if is_valid(previous_position) and get_bit(packed_positions[i], previous_position.row, previous_position.col) == True:
                    last_position = previous_position
                    if d==UP:
                        solution = 'U' + solution
                    elif d==DOWN:
                        solution = 'D' + solution
                    elif d==RIGHT:
                        solution = 'R' + solution
                    elif d==LEFT:
                        solution = 'L' + solution
                    break
    else:
        # If lives>1, it is not possible to walk backwards because
        # some transitions may not be possible. For example, if the
        # actual position is in a green square, some of the neighbors 
        # squares in the previous epoch may have been occupied only with
        # particles with just one life, making impossible to go from 
        # these squares to the actual. When lives>1, it is not enough to
        # have the occupied positions, but also to know the maximum life
        # between all the particles that have been in each position. It 
        # will be necessary a huge amount of memory to do that in the 
        # "forward" propagation, because there is a lot of occupied 
        # positions, with the vast majority representing paths that do 
        # not reach the final position.
        # To drastically reduce the needed memory, we do a "backward" 
        # propagation and store only the lives of the positions that can
        # be reached both at the "forward" and "backward" propagations.
        # These positions are block sparse and represents all the
        # possible minimum paths that goes from the start to the end 
        # position.
        print('Backtracking...')
        start_time_back = time()  
        lives = np.zeros(maze_shape, dtype = np.int8) 
        last_lives = np.zeros((maze_shape[0]+2, maze_shape[1]+2), dtype = np.int8)
        
        sparse_lives_list = [None]*(solution_idx+1)
        total_nonzero = 0
        
        packed_positions = packed_positions[:solution_idx+1, :]
        packed_states = packed_states[:solution_idx+1, :]
        
        if plot_flag == True:
            plt.figure(1)
            plt.clf()
    
        for i in range(solution_idx,-1,-1):
            if (i % iters_per_info == 0):
                print('Epoch {}. Stored positions with nonzero lives: {}'.format(i, total_nonzero))
            states = unpack(packed_states[i])
            
            if i==solution_idx:
                positions = np.full(maze_shape, False)
                positions[last_position.row, last_position.col] = True
                lives[last_position.row, last_position.col] = n_lives
            else:
                lives = np.maximum.reduce([last_lives[1:-1,:-2], last_lives[1:-1,2:] , last_lives[:-2,1:-1], last_lives[2:,1:-1]])
                lives = lives - states
                lives = lives * (lives>0)
                possible_positions = unpack(packed_positions[i])
                lives = lives * possible_positions                
                positions = lives>0
            
            last_lives[1:-1,1:-1] = lives    
            sparse_lives_list[i] = sparse.bsr_matrix(lives, dtype=np.uint8) # TODO: which sparse matrix format is more suitable?
            
            total_nonzero += sparse_lives_list[i].getnnz()
             
            packed_positions[i] = np.packbits(positions)
            
            if plot_flag == True and i % iters_per_plot == 0:
                plot(states+2*(positions+0), fig_idx, 1)
                fig_idx+=1
        
        if plot_flag == True:
            plot(states+2*(positions+0), fig_idx, 1)
            fig_idx+=1

        end_time_back = time()
        print('Backtrack remaining lives: {}'.format(lives[start_pos.row, start_pos.col]))
        print('Elapsed time: {:.2f} s'.format(end_time_back-start_time_back))
    
        # As we did a "backward" propagation, now we can walk forward
        # to find the path
        print('Calculating solution...')
        solution = ''
        last_position = start_pos
        actual_lives = lives[start_pos.row, start_pos.col]
        for i in range(1,solution_idx+1):
            sparse_lives = sparse_lives_list[i].toarray()

            for d in directions():
                next_position = last_position + d
                if is_valid(next_position):
                    last_pos_state = get_bit(packed_states[i-1], last_position.row, last_position.col)
                    
                    next_pos_life = sparse_lives[next_position.row, next_position.col]
                    next_pos_expected_life = actual_lives + (1 if last_pos_state == True else 0)

                    if next_pos_life == next_pos_expected_life:
                        actual_lives = next_pos_expected_life
                        last_position = next_position
                        if d==UP:
                            solution = solution + 'U'
                        elif d==DOWN:
                            solution = solution + 'D'
                        elif d==RIGHT:
                            solution = solution + 'R'
                        elif d==LEFT:
                            solution = solution + 'L'
                        break                               
     
    if len(solution)!=solution_idx:
        raise Exception('Wrong solution size. Expected {}, obtained {}'.format(solution_idx, len(solution)))
    end_time = time()                      
    print('Total elapsed time: {:.2f}'.format(end_time-start_time))
    return solution

def save_solution(solution, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, s in enumerate(solution):
            f.write(s)
            if idx < len(solution)-1:
                f.write(' ')
    print('Solution saved in {}.'.format(filename))


if __name__=='__main__':
    n_challenge = int(sys.argv[1])
    
    if n_challenge<1 or n_challenge>4:
        print('Invalid challenge. Enter a number from 1 to 4.')
    
    image_prefix = 'challenge'+str(n_challenge)+'_img_'
    input_filename = 'input'+str(n_challenge)+'.txt'
    solution_filename = 'output'+str(n_challenge)+'.txt'
    states_filename = 'maze'+str(n_challenge)+'.npy'
    n_lives = 1 if n_challenge !=2 else 6
    exists_unknown_block = False if n_challenge !=4 else True
    
    max_n_epochs = 6500 if n_challenge !=4 else 5000
    
    print('Solving challenge {}...'.format(n_challenge))
    solution = find_shortest_solution(max_n_epochs = max_n_epochs)
    save_solution(solution, solution_filename)
    input('Press any key to terminate.')
