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
    
    def __truediv__(self, den):
        return Position(self.row/den, self.col/den)
        
    def __neg__(self):
        return Position(-self.row, -self.col)
    
    def norm(self):
        return abs(self.row)+abs(self.col) # 1-norma
        

