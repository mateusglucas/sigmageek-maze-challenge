#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from time import time
import os

from position import Position
from automata_maze import AutomataMaze

plot_flag = False
iters_per_info = 10  # iterations interval to print info
iters_per_plot = 100 # iterations interval to plot graph
max_path_len = 50000 # max path length for each particle
min_end_epoch = max_path_len # min epoch for a particle to reach the end position.
                             # limit not applied to the first particle, that will always
                             # have the longest possible path

input_filename = 'input5.txt'
start_solution_filename = '' #'solution5_antigo.txt'
solution_filename = 'output5.txt'
states_filename = 'maze5.npy' 
max_n_epochs = (max_path_len-1)+max_path_len+1 # in the worst case, the first particle 
                                               # makes 50000 movements until it reaches the end. 
                                               # other particle can enter after the 50000-1 movement 
                                               # of the first parcile and make more 50000 movements. 
                                               # The +1 is to consider epoch 0.
        
UP = Position(-1,0)
DOWN = Position(1,0)
LEFT = Position(0,-1)
RIGHT = Position(0,1)


repeller_point = Position(0,0)

np.random.seed(42) # The Answer to the Ultimate Question of Life, The 
                   # Universe, and Everything

def directions(position, aux_cnt):

    delta = position - repeller_point
  
    ret_directions = [] 
     
    correls = np.zeros(4, dtype = int) 
    
    aux_directions = [DOWN, UP, RIGHT, LEFT]
    
    for idx, d in enumerate(aux_directions):
        correls[idx] = d.row*delta.row + d.col*delta.col
        
    correls = np.array(correls)
    for _ in range(4):
        indexes = np.where(correls == correls.max())[0]

        # sorts randomly between the max occurence indexes
        idx = np.random.choice(indexes) 

        ret_directions.append(aux_directions[idx])
        aux_directions.pop(idx)

        correls = np.delete(correls, idx) 
    return ret_directions
 
maze_size = None
maze_shape = None 
end_pos = None
start_pos = None 
n_cols = None     
def get_states(input_filename, states_filename, max_n_epochs):
    global maze_size
    global maze_shape
    global n_cols
    global start_pos
    global end_pos
    
    m = AutomataMaze(input_filename)
    
    maze_size = m.maze.size
    maze_shape = m.maze.shape
    n_cols = maze_shape[1]
    
    start_pos = m.start_position()
    end_pos = m.end_position()
    
    sz = m.maze.size
    sz = sz//8 + (sz % 8 != 0)
    
    if os.path.isfile(states_filename):
        print('Packed states file already exists')
        print('Verifying validity...')
        packed_states = np.load(states_filename)
        if packed_states.shape == (max_n_epochs, sz):
            print('Packed states file valid.')
            return packed_states
        print('Packed states file invalid. Creating new packed states file...')
        
    print('Creating packed states file...')
    packed_states = np.zeros((max_n_epochs, sz), dtype = np.uint8) 
    for i in range(max_n_epochs):
        if i%100 == 0:
            print('Saving iter {}...'.format(i)+' '*10,end='\r')
        packed_states[i] = np.packbits(m.maze)
        m.next()
    print('Saving packed states file...')
    np.save(states_filename, packed_states)
    print('Packed states file saved.')
    return packed_states 

def is_valid(position):
    return position.row>=0 and position.row<maze_shape[0] and position.col>=0 and position.col<maze_shape[1]

def unpack(packed_values):
    return np.unpackbits(packed_values)[:maze_size].reshape(maze_shape) == 1

def get_bit(packed_values, row, col):
    elem_idx = col + row*n_cols
    byte_idx = elem_idx // 8
    bit_idx = 7 - (elem_idx % 8)       

    return (packed_values[byte_idx] & (1<<bit_idx)) != 0 

def set_bit(packed_values, row, col, val = True):
    elem_idx = col + row*n_cols
    byte_idx = elem_idx // 8
    bit_idx = 7 - (elem_idx % 8) 
    
    if val==True:
        packed_values[byte_idx] |= 1<<bit_idx
    else:
        packed_values[byte_idx] &= ~(1<<bit_idx)
    
def find_solution(max_n_epochs):

    max_iter_no_particles = 0
    
    # estados e posições ocupadas
    packed_states = get_states(input_filename, states_filename, max_n_epochs)
    
    start_time = time()
    
    print('Searching solution...')
    
    packed_sz = maze_size//8 + (maze_size % 8 != 0)
    
    max_epoch_to_enter = max_path_len - 1
    
    start_k = 0
    
    loaded_solutions = 0
    n_solutions = 0
    if os.path.isfile(start_solution_filename):
        with open(start_solution_filename, 'r', encoding='utf-8') as f_read:
            with open(solution_filename, 'w', encoding='utf-8') as f_write:
                for line in f_read:
                    f_write.write(line)
                    
                    loaded_solutions+=1
                    n_solutions+=1
                    
                    data = line.replace('\n','').split(' ')

                    k = int(data[0])
                    
                    init_row = start_pos.row
                    init_col = start_pos.col
                     
                    set_bit(packed_states[k], init_row, init_col)
                    
                    solution = data[1:]
                    print('Loading solution {} from epoch {}.'.format(n_solutions, k))
                    
                    solution_idx = k+len(solution)
                    max_epoch_to_enter = max_epoch_to_enter if max_epoch_to_enter<solution_idx-1 else solution_idx-1
                    
                    data_array = np.array(data[1:])
                    d_row = 1*(data_array=='D')-1*(data_array=='U')
                    d_col = 1*(data_array=='R')-1*(data_array=='L')
                    
                    rows = init_row + d_row.cumsum()
                    cols = init_col + d_col.cumsum()
                    
                    for idx, (row, col) in enumerate(zip(rows,cols)):
                        set_bit(packed_states[k+idx+1], row, col)                
        start_k = k+1
        print('Loaded {} solutions. Starting on epoch {}. Elapsed time: {:.2f} s'.format(n_solutions, start_k, time()-start_time))
        start_time = time() # Reset counting after loading partial results
    
    for k in range(start_k, max_n_epochs):
        if k>max_epoch_to_enter:
            print('Last possible epoch. Stopping...')
            break
            
        print('Inserting element in {}-th epoch...'.format(k))
        solution_found = False
        packed_positions = np.zeros((max_n_epochs, packed_sz), dtype = np.uint8)
        
        last_positions = np.full((maze_shape[0]+2, maze_shape[1]+2), False)
        for i in range(k, max_n_epochs):
            if i-k > max_path_len:
                break
            state = unpack(packed_states[i])
            
            if i==k:
                positions = np.full(maze_shape, False)
                positions[start_pos.row, start_pos.col] = True
            else:
                positions = last_positions[:-2,1:-1] | last_positions[2:,1:-1] | last_positions[1:-1,:-2] | last_positions[1:-1,2:]
            positions = positions & (~state)

            if positions.any()==False:
                max_iter_no_particles = max_iter_no_particles if max_iter_no_particles > i-k else i-k
                print('No points remaining (iter {}, max iter {}). Trying next epoch...'.format(i-k, max_iter_no_particles))
                break
            
            # Se a posição final for atingida e for a primeira partícula, armazena a solução temporariamente e segue
            # a busca por soluções mais longas, dentro do limite máximo de movimentos
            # Se não for a primeira partícula, apenas armazena a solução temporariamente se a época atual for
            # menor ou igual ao limite mínimo estipulado de época para término do movimento
            
            # If the final position is reached by the first particle, temporarily store the solution and continue
            # searching for solutions with longer paths, respecting the max movements limit
            # If it isn't the first particle, just store temporarily the solution if the actual epoch
            # is greater or equal to the min epoch to reach the end position  
            if positions[end_pos.row, end_pos.col] == True and (k==0 or min_end_epoch<=i):
                positions[end_pos.row, end_pos.col] = False # remove end position, to continue propagation
                                                            # and avoid solutions in which the particle visits 
                                                            # the end position multiple times
                solution_found = True
                solution_idx = i
                if max_epoch_to_enter<i:
                    break # if it will not reduce the min epoch to enter new particles,
                          # stop the search. Else, continue trying to find a better solution.
            
            packed_positions[i] = np.packbits(positions)
            last_positions[1:-1,1:-1] = positions

            if plot_flag == True and i % iters_per_plot == 0:
                plt.figure(1)
                plt.imshow((state+2*(positions+0)))
                plt.pause(0.001)
        
        if solution_found==True:
            n_solutions+=1
            print('Solution found! End epoch: {}'.format(solution_idx))
            max_epoch_to_enter = max_epoch_to_enter if max_epoch_to_enter<=solution_idx-1 else solution_idx-1

            print('Calculating solution...')
            last_pos = end_pos 
            solution = ''   
            for i in range(solution_idx-1,k-1,-1):              
                added_direction = False
                for d in directions(last_pos, n_solutions):
                    previous_pos = last_pos + d
                    if is_valid(previous_pos) and get_bit(packed_positions[i], previous_pos.row, previous_pos.col) == True:
                        added_direction = True
                        last_pos = previous_pos
                        
                        set_bit(packed_states[i], last_pos.row, last_pos.col)
                        
                        # inverting directions because we are walking
                        # backwards
                        if d==UP:
                            solution = 'D' + solution
                        elif d==DOWN:
                            solution = 'U' + solution
                        elif d==RIGHT:
                            solution = 'L' + solution
                        elif d==LEFT:
                            solution = 'R' + solution
                        break
                if added_direction==False:
                    raise Exception('No direction added') # no reachable statement      
            save_solution(k, solution, solution_filename)
            print('{:.2f} solutions/epoch, total time: {:.2f} s, total solutions: {}, {:.2f} s/solution'.format(n_solutions/(k+1), time()-start_time, n_solutions, (time()-start_time)/(n_solutions-loaded_solutions)))   
        else:
            print('Solution not found.')
            if k==0:
                print('Invalid maze. Particle on epoch 0 should have a solution. Closing...')
                break
    print('Elapsed time: {:.2f} s'.format(time()-start_time))

def save_solution(k, solution, filename):
    file_exists = os.path.isfile(solution_filename)
    with open(filename, 'a', encoding='utf-8') as f:
        if file_exists == True:
            f.write('\n')
        f.write(str(k)+' ')
        for idx, s in enumerate(solution):
            f.write(s)
            if idx < len(solution)-1:
                f.write(' ')
    print('Solution saved in {}.'.format(filename))

if os.path.isfile(solution_filename):
    os.remove(solution_filename)

find_solution(max_n_epochs = max_n_epochs)

input('Press any key to terminate.')
