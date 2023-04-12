#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from time import time

limit_factor = 0.10
plot_flag = True
iters_per_info = 10 # printar info de quantas em quantas iteracoes

input_filename = 'input2.txt'
solution_filename = 'output2.txt' 
n_lifes = 6
exists_unknown_block = False

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
                
def find_solution():

    start_time = time()
    
    m = AutomataMaze(input_filename)
    root = TreeNode(position=m.start_position(), lifes=n_lifes)
    leafs = [root]

    print('Searching for solution...')

    cnt_iter = 0

    nearest = m.min_distance()
    nearest_leaf = root
    
    t = time()
    
    while True:
            
        cnt_iter+=1

        print_info = cnt_iter%iters_per_info == 0
        
        if print_info and plot_flag:
            m.plot()
            for leaf in leafs:
                plt.plot(leaf.position.col, leaf.position.row, 'r.')
            plt.pause(.001)
            plt.clf()
        
        new_leafs = []
        aux_pos = np.full(m.maze.shape, None)
        m.next()

        aux_dict = {}

        for idx_leaf in range(len(leafs)):
            
            # first get even indexes, and after odds. this is done to
            # optimize memory usage when every movement is valid (beginning of challenge 4)
            idx = 2*idx_leaf
            idx = idx if idx<len(leafs) else idx - len(leafs) + (1 if len(leafs)%2 == 0 else 0)
            leaf = leafs[idx]
            
            for direction in directions():
                # verify if new position isn't already in new_leafs
                new_position = leaf.position + direction

                new_pos_dist = (new_position-m.end_position()).norm()

                add_position_flag = False
                        
                # append position if not already in new_leafs and if position is white (False)
                if (limit_factor is None or (new_pos_dist-nearest)/m.min_distance()<limit_factor) and m.is_valid(new_position):
                    if aux_pos[new_position.row, new_position.col] is None:
                        if m.state(new_position) == False:
                            new_leaf = TreeNode(direction = direction, parent = leaf, lifes = leaf.lifes, position = new_position)
                            new_leafs.append(new_leaf)
                            aux_pos[new_position.row, new_position.col] = new_leaf.lifes;
                            add_position_flag = True
                        elif leaf.lifes>1:
                            new_leaf = TreeNode(direction = direction, parent = leaf, lifes = leaf.lifes-1, position = new_position)
                            new_leafs.append(new_leaf)
                            aux_pos[new_position.row, new_position.col] = new_leaf.lifes;
                            add_position_flag = True
                        if add_position_flag == True:           
                            aux_dict[(new_position.row, new_position.col)] = len(new_leafs)-1
                    elif aux_pos[new_position.row, new_position.col]<leaf.lifes:
                        if m.state(new_position) == False:
                            new_leaf = TreeNode(direction = direction, parent = leaf, lifes = leaf.lifes, position = new_position)
                            aux_pos[new_position.row, new_position.col] = new_leaf.lifes;
                            add_position_flag = True
                        elif aux_pos[new_position.row, new_position.col]<leaf.lifes-1:
                            new_leaf = TreeNode(direction = direction, parent = leaf, lifes = leaf.lifes-1, position = new_position)
                            aux_pos[new_position.row, new_position.col] = new_leaf.lifes;
                            add_position_flag = True
                        if add_position_flag == True:
                            new_leafs[aux_dict[(new_position.row, new_position.col)]]=new_leaf
                
                                         
                # if new position was appended
                if add_position_flag==True:
                    # if new position is the closest to the end position
                    if new_pos_dist<nearest:
                        nearest = new_pos_dist
                        nearest_leaf = new_leaf
                        
                    #  if new position is end position, make solution equal to new leaf
                    if new_position == m.end_position():

                        end_time = time()
                        print('Solution found! Total time: {:.2f} s'.format(end_time-start_time)+' '*100)
                        if plot_flag:
                            m.plot()
                            for leaf in new_leafs:
                                plt.plot(leaf.position.col, leaf.position.row, 'r.')
                            plt.pause(.001)
                        
                        return new_leaf
        
        if print_info:
            new_t = time()
            print('Analyzed {} possibilities (iter {}). {:.2f} s/iter. Elapsed time: {:.2f} s '.format(len(leafs), cnt_iter, (new_t-t)/iters_per_info, new_t-start_time), end='')
            print('Nearest {:.2f} ({:.2f}%). '.format(nearest, 100 - nearest/m.min_distance()*100)+' '*20, end='\r')
            t=new_t
        
        # if no solution found, return solution closest to the end point 
        if len(new_leafs)==0:
            end_time = time()
            print('Solution not found, getting nearest solution! Total time: {:.2f} s'.format(end_time-start_time)+' '*100)
            print('Distance to the end: {}'.format(nearest))
            if plot_flag:
                m.plot()
                for leaf in new_leafs:
                    plt.plot(leaf.position.col, leaf.position.row, 'r.')
                plt.pause(.001)
            
            return nearest_leaf
                           
        leafs = new_leafs

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

solution = ''

leaf = find_solution()

while leaf.parent is not None:
    if leaf.direction == UP:
        solution = 'U' + solution
    elif leaf.direction == DOWN:
        solution = 'D' + solution
    elif leaf.direction == LEFT:
        solution = 'L' + solution
    elif leaf.direction == RIGHT:
        solution = 'R' + solution
    leaf = leaf.parent  

save_solution(solution, solution_filename)

#run_solution(solution, 0.25)

input('Press any key to terminate.')
