#!/usr/bin/env python3

import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage
import os
import numpy as np
from position import Position
from automata_maze import AutomataMaze

solution_filename = 'output5.txt'
input_filename = 'input5.txt'
states_filename = 'maze5.npy' 
plot_density = True # plot particles density (for challenge 5)
plot_step = 10
plot_trail = True # plot particles trails or only last particiles positions

def unpack(packed_states, epoch, maze_size, maze_shape):
    return np.unpackbits(packed_states[epoch], count = maze_size).reshape(maze_shape)

m = AutomataMaze(input_filename)

maze_shape = m.maze.shape
maze_size = m.maze.size
n_cols = maze_shape[1]
sz = maze_size//8 + (maze_size % 8 != 0)    

if os.path.isfile(states_filename):
    print('Packed states file found')
    print('Loading...')
    packed_states = np.load(states_filename)
    max_n_epochs = packed_states.shape[0]
    if packed_states.shape[1] == sz:
        print('Packed states with correct size.')
    else:
        print('Packed states with incorrect size. Run solver.py again.')
        exit()

else:
    print('Packed states file not found. Run solver.py.')
    exit()

packed_positions = np.zeros((max_n_epochs, sz), dtype = np.uint8)         

positions = np.full(maze_shape, 0)

with open(solution_filename, 'r', encoding='utf-8') as f:
    for idx_line, line in enumerate(f):
        print('Reading line {}...'.format(idx_line))
        
        data = line.replace('\n','').split(' ')

        if data[0].isnumeric():
            multiple_solutions = True
            k = int(data[0])
        else:
            multiple_solutions = False
            k = 0
        
        init_row = m.start_position().row
        init_col = m.start_position().col
        
        elem_idx = init_col + init_row*n_cols
        byte_idx = elem_idx // 8
        bit_idx = 7 - (elem_idx % 8)   
        packed_positions[k][byte_idx] |= 1<<bit_idx 
        positions[init_row,init_col]+=1
        
        data_array = np.array(data[(1 if multiple_solutions else 0):])
        d_row = 1*(data_array=='D')-1*(data_array=='U')
        d_col = 1*(data_array=='R')-1*(data_array=='L')
        
        rows = init_row + d_row.cumsum()
        cols = init_col + d_col.cumsum()
        
        for idx, (row, col) in enumerate(zip(rows,cols)):
            elem_idx = col + row*n_cols
            byte_idx = elem_idx // 8
            bit_idx = 7 - (elem_idx % 8)       
            packed_positions[k+idx+1][byte_idx] |= 1<<bit_idx 
            positions[row][col]+=1
        
        # Analyze particles density for output with multiple solutions    
        if plot_density==True and multiple_solutions==True:
            if idx_line % plot_step == 0:
                plt.figure(0)
                plt.clf()
                plt.imshow(positions, norm=matplotlib.colors.LogNorm())
                plt.pause(0.01)

if plot_trail==True:
    memo_positions = np.full(maze_shape, False)               

for i in range(0,max_n_epochs):
    states = unpack(packed_states, i, maze_size,maze_shape)
    positions = unpack(packed_positions, i, maze_size, maze_shape)
    
    if plot_trail == True:
        memo_positions = memo_positions | positions
    
    if positions.any()==False:
        break
    
    if i % plot_step==0 or i==max_n_epochs-1:
        plt.figure(1)
        plt.clf()
        if plot_trail==True:
            plt.imshow((states & ~memo_positions) + 2*memo_positions)
        else:
            plt.imshow(states + 2*positions)
        plt.pause(0.01)
    
input('Simulation finished. Press any key to exit...')
        


