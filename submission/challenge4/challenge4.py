#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from time import time
import os

plot_flag = True
iters_per_info = 10 # printar info de quantas em quantas iteracoes
iters_per_plot = 100

input_filename = 'input4.txt'
solution_filename = 'output4.txt'
states_filename = 'maze4.npy' 
n_lifes = 1
exists_unknown_block = True
max_epoch = 4000

class Position:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        
    def __eq__(self, other):
        return self.row == other.row and self.col == other.col
    
    def __add__(self, other):
        return Position(self.row+other.row, self.col+other.col)

    def __sub__(self, other):
        return Position(self.row-other.row, self.col-other.col)
    
    def norm(self):
        return abs(self.row)+abs(self.col) # 1-norma
        
UP = Position(-1,0)
DOWN = Position(1,0)
LEFT = Position(0,-1)
RIGHT = Position(0,1)

def directions():
    return [UP, DOWN, LEFT, RIGHT]

class TreeNode:
    # Initialization by direction and parent or, for the root node, by position
    def __init__(self, direction=None, parent=None, position=None, lifes=1):
        self.parent = parent
        self.direction = direction
        self.lifes = lifes
        self.position = position
        
        if position is None and parent is not None and parent.position is not None and direction is not None:
            self.position = parent.position + direction

class AutomataMaze:
    def __init__(self, filename):
        maze = []
        self.aux_mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        if exists_unknown_block==True:
            unknown_block = []
            with open('block_solution.txt', 'r', encoding='utf-8') as f:
                for row_idx, line in enumerate(f):
                    row = []
                    line = line.replace(' ','').replace('\n','')
                    for col_idx, c in enumerate(line):
                        if c not in ['0', '1']:
                            raise Exception('Invalid character')
                        row.append(c)
                    unknown_block.append(row)
            
        with open(filename, 'r', encoding='utf-8') as f:
            for row_idx, line in enumerate(f):
                row = []
                line = line.replace(' ','').replace('\n','')
                for col_idx, c in enumerate(line):
                    if c not in ['0','1','3','4']:
                        if exists_unknown_block==False or c!='x':
                            raise Exception('Invalid character')
                        elif c=='x' and (row_idx < 2300 or row_idx>2309 or col_idx<2300 or col_idx>2309):
                            raise Exception('Invalid character')
                    
                    row.append(c=='1')
                    if c=='3':
                        self.start_pos = Position(row_idx, col_idx)
                    elif c=='4':
                        self.end_pos = Position(row_idx, col_idx)
                    if exists_unknown_block==True and row_idx >= 2300 and row_idx<=2309 and col_idx>=2300 and col_idx<=2309:
                        expected_char = unknown_block[row_idx-2300][col_idx-2300]
                        if c=='x':
                            row[-1] = expected_char=='1'
                        elif c!=expected_char:
                            print(c, expected_char)
                            raise Exception('Block soluction with incorrect values')
                        
                maze.append(row)
            self.maze = np.array(maze)
            
            self.adjacent_green_cells = ndimage.convolve(self.maze + 0, self.aux_mask, mode='constant')
            
    def min_distance(self):
        return (self.end_pos-self.start_pos).norm()
        
    def plot(self, fig = 0):
        plt.figure(fig)
        plt.imshow(self.maze, cmap='binary')

    def start_position(self):
        return self.start_pos

    def end_position(self):
        return self.end_pos

    def is_valid(self, position):
        return position.row >= 0 and position.row < len(self.maze) and position.col >= 0 and position.col < len(self.maze[0]) 
        
    def state(self, position):
        return self.maze[position.row, position.col]

    def _adjacent_green_cells(self, row_idx, col_idx):
        return self.adjacent_green_cells[row_idx,col_idx]
        
    def next(self):

        white_to_green = (self.adjacent_green_cells>1)*(self.adjacent_green_cells<5) # branco para verde
        green_to_green = (self.adjacent_green_cells>3)*(self.adjacent_green_cells<6) # verde para verde
        is_white = ~self.maze # se é branco
        is_green = self.maze
        
        self.maze = is_green*green_to_green + is_white*white_to_green
        
        # posições final e inicial não possuem estado válido (forçando-as sempre como brancas, vai ter o mesmo efeito)
        self.maze[self.start_pos.row, self.start_pos.col] = False
        self.maze[self.end_pos.row, self.end_pos.col] = False
        
        self.adjacent_green_cells = ndimage.convolve(self.maze + 0, self.aux_mask, mode='constant')

def get_states(input_filename, states_filename, max_epoch):
    m = AutomataMaze(input_filename)
    
    sz = m.maze.size
    sz = sz//8 + (sz % 8 != 0)
    
    if os.path.isfile(states_filename):
        print('Packed states file already exists')
        print('Verifying validity...')
        packed_states = np.load(states_filename)
        if packed_states.shape == (max_epoch, sz):
            print('Packed states file valid.')
            return packed_states
        print('Packed states file invalid. Creating new packed states file...')
        
    print('Creating packed states file...')
    packed_states = np.zeros((max_epoch, sz), dtype = np.uint8) 
    for i in range(max_epoch):
        if i%10 == 0:
            print('Saving iter {}...'.format(i))
        packed_states[i] = np.packbits(m.maze)
        m.next()
    print('Saving packed states file...')
    np.save(states_filename, packed_states)
    print('Packed states file saved.')
    return packed_states 

def get_state(packed_states, epoch, maze_size, maze_shape):
    return np.unpackbits(packed_states[epoch], count = maze_size).reshape(maze_shape)
                
def find_solution(max_epoch):
    packed_states = get_states(input_filename, states_filename, max_epoch)
    
    start_time = time()
    
    print('Searching solution...')
    
    m = AutomataMaze(input_filename)
    start_pos = m.start_position()
    end_pos = m.end_position()
    
    maze_size = m.maze.size
    maze_shape = m.maze.shape
    packed_sz = maze_size//8 + (maze_size % 8 != 0)
    packed_positions = np.zeros((max_epoch, packed_sz), dtype = np.uint8)
    
    b = np.array([[False,True,False],[True,False,True],[False,True,False]])
    
    solution_found = False
    
    for i in range(max_epoch):
        if (i % iters_per_info == 0):
            print('Epoch {}'.format(i))
        state = get_state(packed_states, i, maze_size, maze_shape)

        if i==0:
            positions = np.full(maze_shape, False)
            positions[start_pos.row, start_pos.col] = True
        else:
            positions = ndimage.convolve(last_pos, b, mode = 'constant') 
            positions = positions*(state==0)
        
        packed_positions[i] = np.packbits(positions)
        last_pos = positions
        
        if plot_flag == True and i % iters_per_plot == 0:
            plt.figure(1)
            plt.imshow((state+2*(positions+0)))
            plt.pause(0.001)
        
        if positions[end_pos.row, end_pos.col] == True:
            print('Solution found! (iter {}, total time: {:.2f} s)'.format(i, time()-start_time))
            solution_found = True
            solution_idx = i
            break   
   
    if solution_found:
        print('Calculating solution...')
        last_pos = m.end_position()
    else:
        # find closest solution
        print('Solution not found. Calculating closest solution...')
        
        # armazenado epoca+1, para "nenhuma época" ser representada por 0 e época 0 ser representada por 1
        positions_first_epoch = get_state(packed_positions, max_epoch-1, maze_size, maze_shape) * max_epoch
        for i in range(max_epoch-1-1,-1,-1):
            new_pos = get_state(packed_positions, i, maze_size, maze_shape) > 0
            positions_first_epoch = (positions_first_epoch * (new_pos==0)) + (i+1)*new_pos
        distances = np.zeros(positions_first_epoch.shape)
        for row in range(positions_first_epoch.shape[0]):
            for col in range(positions_first_epoch.shape[1]):
                if positions_first_epoch[row,col]!=0:
                    distances[row,col] = (Position(row,col)-m.end_position()).norm()
        
        distances[distances==0] = distances.max()+1 #evitar que posições inválidas sejam consideradas
         
        min_index = np.unravel_index(distances.argmin(), distances.shape)
        solution_idx = positions_first_epoch[min_index]-1 # subtrair 1, pq é armazenado época+1
        print('Closest distance {} on epoch {}'.format(distances.min(), solution_idx))   
        last_pos = Position(*min_index)
    
    solution = ''   
    for i in range(solution_idx-1,-1,-1):
        positions = get_state(packed_positions, i, maze_size, maze_shape)
        for d in directions():
            previous_pos = last_pos - d
            if m.is_valid(previous_pos) and positions[previous_pos.row, previous_pos.col] == True:
                last_pos = previous_pos
                if d==UP:
                    solution = 'U'+solution
                elif d==DOWN:
                    solution = 'D'+solution
                elif d==RIGHT:
                    solution = 'R'+solution
                elif d==LEFT:
                    solution = 'L'+solution
                break
                              
    return solution
                    
def run_solution(solution, interval=1):
    print('Animating solution...')
    
    m = AutomataMaze(input_filename)
    
    pos = m.start_position()

    if plot_flag:
        plt.figure(1)
        m.plot(1)
        plt.plot(pos.col, pos.row, 'r.')
        plt.pause(interval)

    idx = 0
    while idx<len(solution):
        if solution[idx]=='U':
            pos+=UP
        if solution[idx]=='D':
            pos+=DOWN
        if solution[idx]=='L':
            pos+=LEFT
        if solution[idx]=='R':
            pos+=RIGHT

        m.next()
        
        if plot_flag:
            plt.clf()
            m.plot(1)
            plt.plot(pos.col, pos.row, 'r.')
            plt.pause(interval)

        idx+=1

def save_solution(solution, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, s in enumerate(solution):
            f.write(s)
            if idx < len(solution)-1:
                f.write(' ')
    print('Solution saved in {}.'.format(filename))

solution = find_solution(max_epoch = max_epoch)

save_solution(solution, solution_filename)

#run_solution(solution, 0.25)

input('Press any key to terminate.')
