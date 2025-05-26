# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 73:
# 109625 Francisco Pestana
# 105953 Gonçalo Simplício
from sys import stdin
from search import *
import numpy as np

PIECES = {
    'L': [[0, 1],
          ['X', 1],
          [1, 1]],

    'T': [[1, 1, 1],
          ['X', 1, 'X']],

    'I': [['1', '1', '1', '1']],

    'S': [['X', 1, 1],
          [1, 1, 'X']],
}

#Uma representação abstrata de uma peça
class Piece:
    def __init__(self, id):
        self.id = id
        self.shape = PIECES[id]
        self.variations = self.generate_all_variations()


    def rotate_90(self, shape, k):
        piece_matrix = np.array(shape)
        rotated = np.rot90(piece_matrix, k=k)
        return rotated
    
    def reflect(self, shape):
        piece_matrix = np.array(shape)
        reflected = np.fliplr(piece_matrix)
        return reflected
    

    def generate_all_variations(self):
        variations = set()
        current_shape = np.array(self.shape)
        k = 1
        for _ in range(4):
            shape_tuple = tuple(map(tuple, current_shape))
            variations.add(shape_tuple)
            current_shape = self.rotate_90(current_shape, k)
            k += 1

        k = 1
        current_shape = self.reflect(self.shape)
        for _ in range(4):
            shape_tuple = tuple(map(tuple, current_shape))
            variations.add(shape_tuple)
            current_shape = self.rotate_90(current_shape, k)
            k += 1

        return variations

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

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
        self.blocked_region = None
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
        #Uma forma de guardar os dados das regiões. EXPERIMENTAR
        self.region_values = {}

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
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        #cell = self.cells[row][col]
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
        region = self.get_region(row, col)
        region_cells = self.region_cells(region)
        for cell in region_cells:
            for r, c in directions:
                r, c = cell.row + r, cell.col + c
                if 0 <= r < self.rows and 0 <= c < self.columns and self.cells[r][c].region != cell.region and (r, c) not in Adjacents:
                    Adjacents.append((r, c))
        return Adjacents

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        if 0 < row >= self.rows or 0 < col >= self.columns:
            return []

        adjacent_positions = self.adjacent_positions(row, col)
        neighbours = []
        for r, c in adjacent_positions:
            value = self.cells[r][c].value()
            if value != 'X' not in neighbours:
                neighbours.append(value)
            
        return neighbours
    
    #NOSSSA
    def region_cells(self, region:int, row=None, column=None, visited=None) -> list:
        """Devolve uma lista com todas as celulas de uma região."""
        if visited == None:
            visited = set()
        cells = []

        if row == None or column == None:
            result = self.find_first_region(0, 0, region, None)
            #Na mesma não encontramos
            if result is None:
                return []
            row, column = result
            
        if (row, column) in visited or row < 0 or row >= self.rows  or column < 0 or column >= self.columns:
            return []

        #Evitar visitas/calculos duplicados para certas celas
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
        if row < 0 or row >= self.rows or column < 0 or column >= self.columns:
            return None
        return self.cells[row][column].value()

    #NOSSA
    def get_region(self, row, column):
        return self.cells[row][column].region
    
    #NOSSA
    def number_of_regions(board):
        regions = set()
        for column in board.cells:
            for cell in column:
                regions.add(cell.region)
        return len(regions)
    
    #Nossa
    def region_size(self,region):
        return len(self.region_cells(region))
    
    #Podemos começar com row e column a (0,0)
    #NOSSA
    def find_first_region(self, row, column, region, visited):
        if visited is None:
            visited = set()

        if (row, column) in visited or row < 0 or row >= self.rows  or column < 0 or column >= self.columns:
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
        for row in range(self.rows):
            for column in range(self.columns):
                print(str(self.cells[row][column].value()) + "\t", end="")
            print("\n")

    def _show_board_end_(self):
        """Mostra o tabuleiro no formato final, com as peças colocadas."""
        for row in range(self.rows):
            for column in range(self.columns):
                cell = self.cells[row][column]
                if cell.piece == 'X':
                    # Mostra a região da célula se a peça for 'X'
                    print(str(cell.blocked_region) + "\t", end="")
                else:
                    # Mostra o valor da célula (peça ou região)
                    print(str(cell.value()) + "\t", end="")
            print("\n")

    #Verifica se é possível colocar uma peça no tabuleiro
    #Pode ser usada, ou a inferior
    #ESTA RETORNA A VARIAÇÃO QUE PODE SER COLOCADA
    def can_place_piece(self, piece, start_row, start_col):
        for variation in piece.variations:
            if self.can_place_specific(variation, start_row, start_col, piece.id) and not self.verify_2x2(start_row,start_col):
                return variation
        return None
    
    #Experimenta todas as formas de colocar uma peça
    def can_place_specific(self, variation, start_row, start_col, piece_value):
        region = self.cells[start_row][start_col].region
        adjacent_values = self.adjacent_values(start_row, start_col)

        if self.region_values.get(region, 0) != 0:
            return False #Já existe uma peça nesta região
        
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                row = start_row + i
                col = start_col + j

                if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
                    return False #fora do tabuleiro
                
                cell = self.cells[row][col]
                if value == '1': #Só vasmos considerar os lugares a ser ocupados
                    if cell.piece is not None or cell.region != region or piece_value in adjacent_values or cell.piece == 'X':
                        return False #celula já ocipada
                
                elif value == 'X':
                    if cell.piece is not None:
                        return False
        return True
                    
    def place_piece(self, piece, start_row, start_col):
        for variation in piece.variations:
            if self.can_place_specific(variation, start_row, start_col, piece.id):
                self.place_specific(variation, start_row, start_col, piece.id)
                return True
            
    def place_specific(self, variation, start_row, start_col, piece_value):
        cell_region = self.cells[start_row][start_col].region
        self.region_values[cell_region] = piece_value #Atualiza o valor da região
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                if value == '1' or value == 'X': #Aqui o X determina lugares que não podem ser ocupados (criariamos uma peça 2x2)
                    row = start_row + i
                    col = start_col + j
                    if value == '1': #Parece que se temos carateres e inteiros num array, o numpy transforma tudo a carater
                        self.cells[row][col].piece = piece_value
                    elif value == 'X':
                        self.cells[row][col].piece = 'X'
                        self.cells[row][col].blocked_region = self.cells[row][col].region #Só para mostrar no output final
                        self.cells[row][col].region = None #Esta cela deixa de ser considerada para calculos posteriores
                        
    #Para as regiões de dimensão 4
    def place_piece_dimension_4(self, region):
        pieces = [Piece('L'), Piece('I'), Piece('T'), Piece('S')]
        region_cells = self.region_cells(region)
        for cell in region_cells: #Tecnicamente isto devia ser só uma iteração
            row, col = cell.row, cell.col
            for piece in pieces:
                variation = self.can_place_piece(piece, row, col)
                if variation is not None:
                    self.place_specific(variation, row, col, piece.id)
                    return True

    #sem definir uma coordena, senão definindo uma região
    def try_place_piece_in_region(self, piece, region):
        region_cells = self.region_cells(region)
        for cell in region_cells:
            start_row, start_col = cell.row, cell.col
            if self.can_place_piece(piece, start_row, start_col) != None:
                return True         
        return False  

    # TODO: outros metodos da classe Board

    def all_possibilities(self, region):
        #Devolve todas as peças possíveis por colocar, em todas as suas variações
        pieces = [Piece(piece_id) for piece_id in ['L', 'I', 'T', 'S']]
        possible = []
        region_cells = self.region_cells(region)
        for cell in region_cells:
            for piece in pieces:
                for variation in piece.variations:
                    start_row, start_col = cell.row, cell.col
                    if self.can_place_specific(variation, start_row, start_col, piece.id):
                        possible.append((piece.id, variation, (start_row, start_col)))
        return possible
        
    #NOSSA
    #Devolve um dicionário com as regiões e os valores da peça na região
    def value_regions(self):
        region_values = {}
        region_num = self.number_of_regions()
        for region in range(region_num - 1):
            piece_found = None
            regions_cells = self.region_cells(region + 1) 
            for cell in regions_cells:
                if cell.piece is not None:
                    piece_found = cell.piece
                    break
            region_values[region + 1] = piece_found if piece_found != None else 0
        return region_values
    
    def copy(self):
        # Cria novas células com os mesmos atributos
        new_cells = [[Cell(cell.row, cell.col, cell.region) for cell in row] for row in self.cells]

        # copia também o atributo `piece`:
        for r in range(self.rows):
            for c in range(self.columns):
                new_cells[r][c].piece = self.cells[r][c].piece
                new_cells[r][c].blocked_region = self.cells[r][c].blocked_region

        # Reconstroi as referências entre as células
        for row in range(self.rows):
            for col in range(self.columns):
                cell = new_cells[row][col]
                if row > 0:
                    cell.up = new_cells[row - 1][col]
                if row < self.rows - 1:
                    cell.down = new_cells[row + 1][col]
                if col > 0:
                    cell.left = new_cells[row][col - 1]
                if col < self.columns - 1:
                    cell.right = new_cells[row][col + 1]

        # Cria um novo tabuleiro com as células copiadas
        new_board = Board(new_cells)
        new_board.region_values = dict(self.region_values)  # Copia os valores das regiões
        return new_board
    
    
    #Verifica se existe uma peça 2x2 criada por uma cela especificada
    #True: Existe uma 2x2
    #False: Não Existe uma 2x2
    def verify_2x2(self, row, col):
        piece_values = ['L', 'I', 'T', 'S']
        
        squares = [
        #canto superior esquerdo
        [self.get_value(row, col), self.get_value(row, col + 1),
         self.get_value(row + 1, col), self.get_value(row + 1, col + 1)],  

        #Superior direito
        [self.get_value(row, col), self.get_value(row, col - 1),
         self.get_value(row + 1, col), self.get_value(row + 1, col - 1)],  

        #Inferior esquerdo
        [self.get_value(row, col), self.get_value(row, col + 1),
         self.get_value(row - 1, col), self.get_value(row - 1, col + 1)],  
         
        #Inferior direito
        [self.get_value(row, col), self.get_value(row, col - 1),
         self.get_value(row - 1, col), self.get_value(row - 1, col - 1)]   
        ]

        for square in squares:
            count = 0
            for value in square:
                if value in piece_values:
                    count += 1
            if count == 4:
                return True #Existe uma peça 2x2
            
        return False #Não existe uma peça 2x2

class Nuruomino(Problem):
    state_id = 0 
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        self.initial = NuruominoState(board)
        self.regions = board.value_regions()

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        if state is None:
            return []
        
        actions = list()
        current_regions = state.board.value_regions()
        for region, item in current_regions.items():
            if item == 0:
                possibilities = state.board.all_possibilities(region)
                for piece_id, variation, (row, col) in possibilities:
                    if state.board.can_place_specific(variation, row, col, piece_id):
                        actions.append((Piece(piece_id), variation, row, col))
        return actions

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""    
        piece, variation, row, col = action
        new_board = state.board.copy()  # Cria uma cópia do tabuleiro atual

        if new_board.can_place_specific(variation, row, col, piece.id):
            new_board.place_specific(variation, row, col, piece.id)

            for i in range(state.board.rows - 1):
                for j in range(state.board.columns - 1):
                    square = [
                        new_board.get_value(i, j),
                        new_board.get_value(i, j+1),
                        new_board.get_value(i+1, j),
                        new_board.get_value(i+1, j+1)
                    ]
                    # conta todas as células “ocupadas” (peça ≠ None e ≠ 'X')
                    if sum(1 for v in square if v in ['L','I','T','S']) == 4:
                        return None
                    
            new_board.region_values = new_board.value_regions() 
            sucessor = NuruominoState(new_board)
            return sucessor 
        
        return None

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""

        if state is None:
            return False

        current_regions = state.board.value_regions()

        for i in range(state.board.rows - 1):
            for j in range(state.board.columns - 1):
                square = [
                    state.board.get_value(i, j),
                    state.board.get_value(i, j+1),
                    state.board.get_value(i+1, j),
                    state.board.get_value(i+1, j+1)
                ]
                # conta todas as células “ocupadas”
                if sum(1 for v in square if v in ['L','I','T','S']) == 4:
                    return False

        for region, value in current_regions.items():
            if value == 0:
                return False
            row, col = state.board.find_first_region(0, 0, region, None)
            adjacent_values = state.board.adjacent_values(row, col)
            if value in adjacent_values:
                return False
        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        len(self.regions) - count(self.regions ==0)

if __name__ == "__main__":
    #TEST DO NURUOMINO__________________________________________________________________
    board = Board.parse_instance()
    board._show_board_()

    for region in range(board.number_of_regions()):
        if board.region_size(region + 1) == 4:
            board.place_piece_dimension_4(region + 1)
    board.region_values = board.value_regions()
    
    print("As de dimensão 4 já foram\n")

    board._show_board_()
    
    print("SEPARATOR\n")
    
    problem = Nuruomino(board)

    #solution = depth_first_graph_search(problem)
    solution = astar_search(problem,problem.h(),True)
    # Mostra o resultado
    if solution:
        solution.state.board._show_board_end_()
    else:
        print("Nenhuma solução encontrada")