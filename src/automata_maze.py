import numpy as np
from matplotlib import pyplot as plt
from position import Position

# TODO: otimizar para remover convolução, como fiz no solver

class AutomataMaze:
    def __init__(self, filename):
        maze = []
        self.neighbors_mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
                   
        with open(filename, 'r', encoding='utf-8') as f:
            for row_idx, line in enumerate(f):
                row = []
                line = line.replace(' ','').replace('\n','')
                for col_idx, c in enumerate(line):
                    if c not in ['0','1','3','4','x']:
                        raise Exception('Invalid character')

                    row.append(c=='1') # 'x', '3' e '4' ficam como células mortas.
                                       # É tarefa do usuário substituir o bloco com
                                       # 'x' usando a função modify_maze 
                    
                    if c=='3':
                        self.start_pos = Position(row_idx, col_idx)
                    elif c=='4':
                        self.end_pos = Position(row_idx, col_idx)                    
                maze.append(row)
            self.maze = np.array(maze)
            
            maze_shape = self.maze.shape
            aux = np.zeros((maze_shape[0]+2, maze_shape[1]+2), dtype=np.uint8)
            aux[1:-1,1:-1] = self.maze
            self.adjacent_green_cells = aux[:-2,1:-1] + aux[2:,1:-1] + aux[1:-1,2:] + aux[1:-1,:-2] + aux[:-2,:-2] + aux[2:,2:] + aux[2:,:-2] + aux[:-2,2:] 
    
    def modify_maze(self, new_block, row_to_insert, col_to_insert):
        n_rows = new_block.shape[0]
        n_cols = new_block.shape[1]
        self.maze[row_to_insert:row_to_insert + n_rows, col_to_insert:col_to_insert+n_cols] = new_block
        
        maze_shape = self.maze.shape
        aux = np.zeros((maze_shape[0]+2, maze_shape[1]+2), dtype=np.uint8)
        aux[1:-1,1:-1] = self.maze
        self.adjacent_green_cells = aux[:-2,1:-1] + aux[2:,1:-1] + aux[1:-1,2:] + aux[1:-1,:-2] + aux[:-2,:-2] + aux[2:,2:] + aux[2:,:-2] + aux[:-2,2:] 
              
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
       
    def next(self):

        white_to_green = (self.adjacent_green_cells>1)*(self.adjacent_green_cells<5) # branco para verde
        green_to_green = (self.adjacent_green_cells>3)*(self.adjacent_green_cells<6) # verde para verde
        is_white = ~self.maze # se é branco
        is_green = self.maze
        
        self.maze = is_green*green_to_green + is_white*white_to_green
        
        # posições final e inicial não possuem estado válido (forçando-as sempre como brancas, vai ter o mesmo efeito)
        self.maze[self.start_pos.row, self.start_pos.col] = False
        self.maze[self.end_pos.row, self.end_pos.col] = False
        
        maze_shape = self.maze.shape
        aux = np.zeros((maze_shape[0]+2, maze_shape[1]+2), dtype=np.uint8)
        aux[1:-1,1:-1] = self.maze
        self.adjacent_green_cells = aux[:-2,1:-1] + aux[2:,1:-1] + aux[1:-1,2:] + aux[1:-1,:-2] + aux[:-2,:-2] + aux[2:,2:] + aux[2:,:-2] + aux[:-2,2:] 
