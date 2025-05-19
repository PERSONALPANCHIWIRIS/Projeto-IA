# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2
from sys import stdin
from search import *

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = Nuruomino.state_id
        Nuruomino.state_id += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

#Cada coordenada do tabuleiro
class Cell:
    def __init__(self, row, col, region):
        self.row = row
        self.col = col
        self.region = region
        self.piece = None
        #Ligações para os espaços adjacentes
        self.up = None
        self.down = None
        self.left = None
        self.right = None

    def value(self):
        return self.piece if self.piece != None else self.region



class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    def __init__(self, cells):
        self.cells = cells
        self.rows = len(cells)
        self.columns = len(cells)

    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        cells = self.region_cells(region)
        neighbours = set()

        directions = [  
            (-1,  0),  # cima
            ( 1,  0),  # baixo
            ( 0, -1),  # esquerda
            ( 0,  1),  # direita
            (-1, -1),  # cima esquerda
            (-1,  1),  # cima direita
            ( 1, -1),  # baixo esquerda
            ( 1,  1)   # baixo direita
        ]

        for cell in cells:
            for r, c in directions:
                r, c = cell.row + r, cell.col + c
                if 0 <= r < self.rows and 0 <= c < self.columns:
                    neighbor = self.cells[r][c]
                    if neighbor.region != region:
                        neighbours.add(neighbor.region)

        return list(neighbours)

        #TODO
        pass
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        cell = self.cells[row][col]
        if 0 > row >= self.rows or 0 > col >= self.columns:
            return []
        directions = [  
            (-1, -1),  # cima esquerda
            (-1,  0),  # cima
            (-1,  1),  # cima direita
            ( 0, -1),  # esquerda    
            ( 0,  1),  # direita   
            ( 1, -1),  # baixo esquerda  
            ( 1,  0),  # baixo
            ( 1,  1),   # baixo direita
              
        ]
        Adjacents = []
        for dir in directions:
            r, c = cell.row + dir[0], cell.col + dir[1]
            if 0 <= r < self.rows and 0 <= c < self.columns:
                Adjacents.append((r, c))
        return Adjacents
        #TODO
        pass

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        if 0 < row >= self.rows or 0 < col >= self.columns:
            return []
        
        directions = [  
            (-1, -1),  # cima esquerda
            (-1,  0),  # cima
            (-1,  1),  # cima direita
            ( 0, -1),  # esquerda    
            ( 0,  1),  # direita   
            ( 1, -1),  # baixo esquerda  
            ( 1,  0),  # baixo
            ( 1,  1),   # baixo direita
              
        ]
        cell = self.cells[row][col]
        neighbours = []
        for r, c in directions:
            r, c = cell.row + r, cell.col + c
            if 0 <= r < self.rows and 0 <= c < self.columns:
                neighbours.append(self.cells[r][c].value())

        return neighbours
            
        #TODO
        pass
    
    #NOSSSA
    def region_cells(self, region:int, row=None, column=None, visited=None) -> list:
        """Devolve uma lista com todas as celulas de uma região."""
        if visited == None:
            visited = set()
        cells = []

        if row == None or column == None:
            (row, column) = self.find_first_region(0, 0, region, None)
            if row == None or column == None:
                return []
            
        if (row, column) in visited or row < 0 or row >= self.rows  or column < 0 or column >= self.columns:
            return []

        visited.add((row, column))

        if self.cells[row][column].region == region:
            cells.append(self.cells[row][column])

        cells += self.region_cells(region, row + 1, column, visited)
        cells += self.region_cells(region, row - 1, column, visited)
        cells += self.region_cells(region, row, column + 1, visited)
        cells += self.region_cells(region, row, column - 1, visited)

        return cells

    #Como o nome indica
    def get_value(self, row, column):
        print(str(self.cells[row][column].value()))

    #NOSSA
    def get_region(self, row, column):
        print(str(self.cells[row][column].region))
    
    #Podemos começar com row e column a (0,0)
    #NOSSA
    def find_first_region(self, row, column, region, visited):
        if visited is None:
            visited = set()

        if (row, column) in visited or row < 0 or row > self.rows  or column < 0 or column > self.columns:
            return None

        visited.add((row, column))

        if self.cells[row][column].region == region:
            return (row, column)
        
        right = self.find_first_region(row, column + 1, region, visited)
        if right != None:
            return right
        down = self.find_first_region(row + 1, column, region, visited)
        if down != None:
            return down
        
        #Não existe região
        return None


    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        #Separamos as linhas
        lines = stdin.read().strip().splitlines()
        #Cada linha é separada nos seus valores
        matrix = [[int(value) for value in line.split()] for line in lines]
        cells = [[Cell(row, column, matrix[row][column]) for column in
        range(len(matrix[row]))] for row in range(len(matrix))]
        #Vai criando as diferentes celulas do tabuleiro
        for row in range(len(matrix)):
            for column in range(len(matrix[row])):
                cell = cells[row][column]
                if row > 0:
                    cell.up = matrix[row-1][column]
                if row != len(matrix) - 1:
                    cell.down = matrix[row+1][column]
                if column > 0:
                    cell.left = matrix[row][column - 1]
                if column != len(matrix) - 1:
                    cell.right = matrix[row][column + 1]
        return Board(cells)
    
    def _show_board_(self):
        #print("\n")
        for row in range(self.rows):
            for column in range(self.columns):
                print(str(self.cells[row][column].value()) + " ", end="")
            print("\t")

        #TODO
        pass    

    # TODO: outros metodos da classe Board

class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        #TODO
        pass 

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #TODO
        pass 

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        #TODO
        pass 
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass


#TESTES
if __name__ == "__main__":
    board = Board.parse_instance()
    board._show_board_()
    board.get_value(0, 0)
    board.get_value(2,2)
    print(str(board.find_first_region(0, 0, 1, None)))
    print(str(board.find_first_region(0, 0, 3, None)))
    print(board.adjacent_regions(1))
    print(board.adjacent_values(1, 4))
    print(board.adjacent_positions(3, 3))