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

        #TODO
        pass
    
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
        # for dir in directions:
        #     r, c = cell.row + dir[0], cell.col + dir[1]
        #     if 0 <= r < self.rows and 0 <= c < self.columns:
        #         Adjacents.append((r, c))
        # return Adjacents
        region = self.get_region(row, col)
        region_cells = self.region_cells(region)
        for cell in region_cells:
            for r, c in directions:
                r, c = cell.row + r, cell.col + c
                if 0 <= r < self.rows and 0 <= c < self.columns and self.cells[r][c].region != cell.region and (r, c) not in Adjacents:
                    Adjacents.append((r, c))
        return Adjacents
        #TODO
        pass

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        if 0 < row >= self.rows or 0 < col >= self.columns:
            return []
        
        # directions = [  
        #     (-1, -1),  # cima esquerda
        #     (-1,  0),  # cima
        #     (-1,  1),  # cima direita
        #     ( 0, -1),  # esquerda    
        #     ( 0,  1),  # direita   
        #     ( 1, -1),  # baixo esquerda  
        #     ( 1,  0),  # baixo
        #     ( 1,  1),   # baixo direita
              
        # ]
        #cell = self.cells[row][col]
        # region = self.get_region(row, col)
        # region_cells = self.region_cells(region)
        # neighbours = []
        # for cell in region_cells:
        #     for r, c in directions:
        #         r, c = cell.row + r, cell.col + c
        #         if 0 <= r < self.rows and 0 <= c < self.columns and self.cells[r][c].region != region:
        #             #Não sei se queremos só os valores para verificação, ou literalmente cada valor de cada cela adjacente à região
        #             if self.cells[r][c].value() not in neighbours:
        #                 neighbours.append(self.cells[r][c].value())
        # return neighbours

        adjacent_positions = self.adjacent_positions(row, col)
        neighbours = []
        for r, c in adjacent_positions:
            value = self.cells[r][c].value()
            if value != 'X' not in neighbours:
                neighbours.append(value)
            
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
            #Na mesma não encontramos
            if row == None or column == None:
                return []
            
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
        #print("\n")
        for row in range(self.rows):
            for column in range(self.columns):
                print(str(self.cells[row][column].value()) + " ", end="")
            print("\t")

        #TODO
        pass    
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
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                row = start_row + i
                col = start_col + j

                if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
                    #print("fora do tabuleiro")
                    return False #fora do tabuleiro

                cell = self.cells[row][col]
                if value == '1': #Só vasmos considerar os lugares a ser ocupados
                    # row = start_row + i
                    # col = start_col + j
                    #print("row: " + str(row) + " col: " + str(col))
                    # if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
                    #     #print("fora do tabuleiro")
                    #     return False #fora do tabuleiro
                    # cell = self.cells[row][col]
                    #adjacent_values = self.adjacent_values(row, col)
                    # print((row, col))
                    # print(piece_value)
                    # print(adjacent_values)
                    #print("Defined:" + str(region))
                    #print("Cell:" + str(cell.region))
                    #if value == 1:
                    if cell.piece is not None or cell.region != region or piece_value in adjacent_values or cell.piece == 'X':
                        #print("ja ocupada ou diferente, ou ao lado temos uma igual")
                        return False #celula já ocipada
                    # elif value == 'X':
                    #     if cell.piece is not None:
                    #         return False
                
                elif value == 'X':
                    if cell.piece is not None:
                        return False
                    
        #print(variation)
        #print("CAN PLACE\n")
        return True
                    
    def place_piece(self, piece, start_row, start_col):
        for variation in piece.variations:
            #print("PRE CAN PLACE\n")
            if self.can_place_specific(variation, start_row, start_col, piece.id):
                #print("POST CAN PLACE\n")
                self.place_specific(variation, start_row, start_col, piece.id)
                #print("POST PLACE\n")
                return True
            
    def place_specific(self, variation, start_row, start_col, piece_value):
        #TESTAR
        cell_region = self.cells[start_row][start_col].region
        self.region_values[cell_region] = piece_value #Atualiza o valor da região
        #TESTAR
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                if value == '1' or value == 'X': #Aqui o X determina lugares que não podem ser ocupados (criariamos uma peça 2x2)
                    row = start_row + i
                    col = start_col + j
                    if value == '1': #Parece que se temos carateres e inteiros num array, o numpy transforma tudo a carater
                        self.cells[row][col].piece = piece_value
                    elif value == 'X':
                        self.cells[row][col].piece = 'X'
                        self.cells[row][col].region = None #Esta cela deixa de ser considerada para calculos posteriores
                    #print("VALUE:" + str(piece_value) + "\n")

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
                #print("PODE PODE PODE\n")
                return True         
        #print("NÃO PODE PODE PODE\n")
        return False  

    # TODO: outros metodos da classe Board

    def all_possibilities(self, region):
        #Devolve todas as peças possíveis por colocar, em todas as suas variações
        pieces = [Piece(piece_id) for piece_id in ['L', 'I', 'T', 'S']]
        possible = []
        region_cells = self.region_cells(region)
        for cell in region_cells:
            start_row, start_col = cell.row, cell.col
            for piece in pieces:
                variation = self.can_place_piece(piece, start_row, start_col)
                if variation is not None:
                    possible.append((piece.id, variation, (start_row, start_col)))
        return possible
        
    #NOSSA
    #Devolve um dicionário com as regiões e os valores da peça na região
    #TEMPORARIA; EXPERIMENTAR OUTRAS FORMAS
    def value_regions(self):
        region_values = {}
        region_num = self.number_of_regions()
        for region in range(region_num - 1):
            #print("REGIÃO: " + str(region + 1))
            piece_found = None
            regions_cells = self.region_cells(region + 1) 
            for cell in regions_cells:
                if cell.piece is not None:
                    piece_found = cell.piece
                    #region_values[cell.region] = cell.piece if cell.piece
                    break
            region_values[region + 1] = piece_found if piece_found != None else 0
        return region_values
    
    
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
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        #self.regions = board.number_of_regions()
        #self.regions = np.zeros(board.number_of_regions(), dtype=object)
        self.regions = board.value_regions()
        #TODO
        pass 

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = list()
        for region in self.regions.items() == 0:
            actions.append(self.board.all_possibilities(region))
            # pieces = [Piece(piece_id) for piece_id in ['L','T','I','S']]
            # for piece in pieces:
            #     value = self.board.try_place_piece_in_region(piece,region)
            #     if value:
            #         actions.append([region,piece.id,piece.shape])
        return actions
        #TODO
        pass 

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        if action in self.actions(state):
            self.board.place_piece(action.piece)
            self.regions[action[0]] = action[1]
            state.board = self.board
            return state
        #TODO
        pass 
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        if len(self.regions) != self.board.number_of_regions():
            return False
        
        #Isto aqui é usando as funções do tabuleiro
        for region in self.regions.keys():
            row, col = self.board.find_first_region(0, 0, region, None)
            adjacent_values = self.board.adjacent_values(row, col)
            if self.regions[region] == 0 or self.regions[region] in adjacent_values:
                return False
        return True

        #aqui seria usando o numpy array
        # for region in range(1, len(self.regions)):
        #     if self.regions[region - 1] == 0:
        #         return False
        #     adjacents = self.board.adjacent_regions(region)
        #     for adj in adjacents:
        #         if self.regions[region - 1] == self.regions[adj - 1]:
        #             return False
        # return True
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # placed =0 
        # for region in len(self.regions):
        #     if self.regions[region] == 0:
        #         placed+1
        len(self.regions) - count(self.regions ==0)
        node.action.
        # TODO
        pass


#TESTES
if __name__ == "__main__":
    board = Board.parse_instance()
    board._show_board_()

    print("Values de coordenadas especificas\n")
    board.get_value(0, 0)
    board.get_value(2,2)

    print("Região de uma coordenada especifica\n")
    board.get_region(1, 1)
    board.get_region(3, 4)

    print("Find first in region\n")
    print(str(board.find_first_region(0, 0, 1, None)))
    print(str(board.find_first_region(0, 0, 3, None)))

    print("Todas as cellas de uma região\n")
    cells_1 = board.region_cells(1)
    print("Região 1\n")
    for cell in cells_1:
        print(f"({cell.row}, {cell.col})")
    cells_2 = board.region_cells(2)
    print("Região 2\n")
    for cell in cells_2:
        print(f"({cell.row}, {cell.col})")

    print("regiões adjacentes a uma especifica\n")
    print(board.adjacent_regions(1))
    print(board.adjacent_regions(3))
    print(board.adjacent_regions(5))

    print("Valores adjacentes a uma região\n")
    print(board.adjacent_values(1, 4))
    print(board.adjacent_values(5,5))

    print("Coordenadas adjacentes a uma região\n")
    print(board.adjacent_positions(3, 3))
    print(board.adjacent_positions(0, 0))

    print("Vem as peças:\n")
    L_piece = Piece('L')
    S_piece = Piece('S')
    T_piece = Piece('T')
    I_piece = Piece('I')
    print(L_piece.id)
    print(L_piece.variations)
    print(S_piece.id)
    print(S_piece.variations)
    print(T_piece.id)
    print(T_piece.variations)
    print(I_piece.id)
    print(I_piece.variations)

    #Tentar colocar L na região 1
    print("Tentar colocar L na região 1\n")
    board.place_piece(L_piece, 0, 0)

    print("Depois de colocar L na região 1\n")
    board._show_board_()

    print("Região 2 deve ter menos uma cela\n")
    cells_2 = board.region_cells(2)
    print("Região 2\n")
    for cell in cells_2:
        print(f"({cell.row}, {cell.col})")

    print(board.get_value(1, 1))

    print("Tenta colocar I na região 5\n")
    board.place_piece(I_piece, 2, 5)
    print("Depois de colocar I na região 5\n")
    board._show_board_()

    print("tentar colocar I na região 2\n")
    board.place_piece(I_piece, 0, 2)
    board._show_board_()

    print("Indica se pode ao não colocar\n")
    board.try_place_piece_in_region(I_piece, 2)
    board.try_place_piece_in_region(T_piece, 2)



    # board = Board.parse_instance()
    # board._show_board_()
    # problem = Nuruomino(board)
    # for region in problem.regions.keys:
    #     if board.region_size(region) == 4:
    #         problem.board.place_piece_dimension_4(region)
    # problem.regions = problem.board.value_regions()
    # node= depth_first_tree_search(problem)
    # if problem.goal_test(node):
    #     problem.board._show_board_()