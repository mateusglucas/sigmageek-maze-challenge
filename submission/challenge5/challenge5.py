#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from time import time
import cv2
import os

plot_flag = False
iters_per_info = 10  # printar info de quantas em quantas iteracoes
iters_per_plot = 100 # plotar gráficos de quantas em quantas iteracoes
max_path_len = 25000 # tamanho máximo do caminho para cada partícula. 
                     # utilizado 25000 devido ao tempo de processamento 
                     # e prazo de entrega. O valor correto é 50000

input_filename = 'input5.txt'
start_solution_filename = 'output5_antigo.txt'
solution_filename = 'output5.txt'
states_filename = 'maze5.npy' 
max_n_epochs = (max_path_len-1)+max_path_len+1 # no pior dos casos, a primeira partícula 
                                               # faz 50000 movimentos até chegar no fim. 
                                               # outra partícula pode entrar após o 50000-1
                                               # movimento da primeira partícula e realizar
                                               # mais 50000 movimentos. O +1 é para considera
                                               # a época 0.

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
            
        with open(filename, 'r', encoding='utf-8') as f:
            for row_idx, line in enumerate(f):
                row = []
                line = line.replace(' ','').replace('\n','')
                for col_idx, c in enumerate(line):
                    if c not in ['0','1','3','4']:
                        raise Exception('Invalid character')
                    
                    row.append(c=='1')
                    if c=='3':
                        self.start_pos = Position(row_idx, col_idx)
                    elif c=='4':
                        self.end_pos = Position(row_idx, col_idx)
                        
                maze.append(row)
            self.maze = np.array(maze)
            
            self.adjacent_green_cells = cv2.filter2D(np.array(self.maze,dtype=float), -1, self.aux_mask, borderType=cv2.BORDER_ISOLATED)
            self.adjacent_green_cells = np.array(self.adjacent_green_cells.round(), dtype=int)
            
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
        
        self.adjacent_green_cells = cv2.filter2D(np.array(self.maze, dtype=float), -1, self.aux_mask, borderType=cv2.BORDER_ISOLATED)
        self.adjacent_green_cells = np.array(self.adjacent_green_cells.round(), dtype=int)

def get_states(input_filename, states_filename, max_n_epochs):
    m = AutomataMaze(input_filename)
    
    sz = m.maze.size
    sz = sz//8 + (sz % 8 != 0)
    
    if os.path.isfile(states_filename):
        print('Maze epochs file already exists')
        print('Verifying dimensions...')
        packed_states = np.load(states_filename)
        if packed_states.shape == (max_n_epochs, sz):
            print('Dimensions are valid.')
            return packed_states
        print('Dimensions invalid. Creating new maze epochs...')
        
    print('Creating maze epochs file...')
    packed_states = np.zeros((max_n_epochs, sz), dtype = np.uint8) 
    for i in range(max_n_epochs):
        if i%100 == 0:
            print('Saving epoch {}...'.format(i)+' '*10,end='\r')
        packed_states[i] = np.packbits(m.maze)
        m.next()
    print('Saving maze epochs file...')
    np.save(states_filename, packed_states)
    print('Maze epochs file saved.')
    return packed_states 

def get_state(packed_states, epoch, maze_size, maze_shape):
    return np.unpackbits(packed_states[epoch], count = maze_size).reshape(maze_shape)
                
def find_solution(max_n_epochs):
    packed_states = get_states(input_filename, states_filename, max_n_epochs)
    
    start_time = time()
    
    print('Searching solution...')
    
    m = AutomataMaze(input_filename)
    start_pos = m.start_position()
    end_pos = m.end_position()
    
    maze_size = m.maze.size
    maze_shape = m.maze.shape
    packed_sz = maze_size//8 + (maze_size % 8 != 0)

    packed_occupied_pos = np.zeros((max_n_epochs, packed_sz), dtype = np.uint8) # posições ocupadas pelas soluções passadas
    
    b = np.array([[0,1,0],[1,0,1],[0,1,0]])
    
    solutions = []
    
    max_epoch_to_enter = max_path_len - 1
    
    start_k = 0
    
    if os.path.isfile(start_solution_filename):
        print('File with solutions found. Loading solutions...')
        with open(start_solution_filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.replace('\n','').split(' ')

                k = int(data[0])
                
                row = m.start_position().row
                col = m.start_position().col
                
                positions = np.full(maze_shape, False)
                positions[row, col] = True
                
                old_packed_occupied_positions = get_state(packed_occupied_pos, k, maze_size, maze_shape)
                new_positions = positions + old_packed_occupied_positions
                packed_occupied_pos[k] = np.packbits(new_positions)
                
                solution = data[1:]
                solutions.append((k, solution))
                print('Loading solution for epoch {}'.format(k))
                
                save_solution(k, solution, solution_filename)
                solution_idx = k+len(solution)
                max_epoch_to_enter = max_epoch_to_enter if max_epoch_to_enter<=solution_idx-1 else solution_idx-1
                
                for idx, d in enumerate(data[1:]):
                    if d=='R':
                        col+=1
                    elif d=='U':
                        row-=1
                    elif d=='D':
                        row+=1
                    elif d=='L':
                        col-=1
                    else:
                        print('Found unexpected character')
                        break
                    positions = np.full(maze_shape, False)
                    positions[row, col] = True
                    
                    old_packed_occupied_positions = get_state(packed_occupied_pos, k+idx+1, maze_size, maze_shape)
                    new_positions = positions + old_packed_occupied_positions
                    packed_occupied_pos[k+idx+1] = np.packbits(new_positions)
                    
        start_k = k+1
        print('Loaded {} solutions. Starting search on epoch {}'.format(len(solutions), start_k))
    
    old_packed_positions = get_state(packed_occupied_pos, 400, maze_size, maze_shape)
    
    for k in range(start_k, max_n_epochs):
        if k>max_epoch_to_enter:
            print('Last possible epoch. Stopping...')
            break
            
        print('Trying to inserting particle in epoch {}...'.format(k))
        solution_found = False
        packed_positions = np.zeros((max_n_epochs, packed_sz), dtype = np.uint8)
        
        for i in range(k, max_n_epochs):
            if i-k > max_path_len:
                break
            state = get_state(packed_states, i, maze_size, maze_shape)
            occupied_pos = get_state(packed_occupied_pos, i, maze_size, maze_shape)
            
            if i==k:
                positions = np.full(maze_shape, False)
                positions[start_pos.row, start_pos.col] = True
            else:
                positions = cv2.filter2D(np.array(last_positions, dtype=float), -1, b, borderType = cv2.BORDER_ISOLATED).round() > 0
                positions = positions*(state==0)
            positions = positions*(occupied_pos==0)

            if positions.any()==False:
                print('No points remaining. Trying next epoch...')
                break
            
            if positions[end_pos.row, end_pos.col] == True:
                positions[end_pos.row, end_pos.col] = False
                solution_found = True
                solution_idx = i
                if max_epoch_to_enter<=i-1:
                    break # se não vai reduzir a época limite de entrada de novas partículas, encerrar busca.
                          # caso contrário, seguir tentando achar uma solução melhor.
            
            packed_positions[i] = np.packbits(positions)
            last_positions = positions

            if plot_flag == True and i % iters_per_plot == 0:
                plt.figure(1)
                plt.imshow((state+2*(positions+0)))
                plt.pause(0.001)
             
        if solution_found==True:
            print('Solution found! (epoch {}, total time: {:.2f} s, total solutions: {})'.format(solution_idx, time()-start_time, len(solutions)+1))
            max_epoch_to_enter = max_epoch_to_enter if max_epoch_to_enter<=solution_idx-1 else solution_idx-1
            print('Calculating solution...')
            last_pos = m.end_position()   
            solution = ''   
            for i in range(solution_idx-1,k-1,-1):
                positions = get_state(packed_positions, i, maze_size, maze_shape)
                
                occupied_pos = np.full(maze_shape, False)
                
                old_occupied_pos = get_state(packed_occupied_pos, i, maze_size, maze_shape)
                
                added_direction = False
                for d in directions():
                    previous_pos = last_pos - d
                    if m.is_valid(previous_pos) and positions[previous_pos.row, previous_pos.col] == True:
                        added_direction = True
                        occupied_pos[previous_pos.row, previous_pos.col] = True
                        new_occupied_pos = old_occupied_pos + occupied_pos
                        
                        packed_occupied_pos[i] = np.packbits(new_occupied_pos)
                        
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
                if added_direction==False:
                    raise Exception('No direction added') # se chegou aqui, algo deu errado
                
            solutions.append((k, solution))
            save_solution(k, solution, solution_filename)
        else:
            print('Solution not found.')
    return solutions

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

solutions = find_solution(max_n_epochs = max_n_epochs)

#save_solutions(solutions, solution_filename)

input('Press any key to terminate.')
