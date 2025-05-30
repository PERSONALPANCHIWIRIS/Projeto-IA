# nuruomino_optimized.py: Versão otimizada do projeto Nuruomino
# Grupo 73: 109625 Francisco Pestana, 105953 Gonçalo Simplício

from sys import stdin
from search import *
import numpy as np
from collections import deque

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

class Piece:
    _variations_cache = {}
    
    def __init__(self, id):
        self.id = id
        self.shape = PIECES[id]
        
        # Cache das variações para evitar recalcular
        if id not in Piece._variations_cache:
            Piece._variations_cache[id] = self.generate_all_variations()
        self.variations = Piece._variations_cache[id]

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
        
        # Rotações
        for k in range(4):
            shape_tuple = tuple(map(tuple, current_shape))
            variations.add(shape_tuple)
            current_shape = self.rotate_90(current_shape, 1)

        # Reflexões + rotações
        current_shape = self.reflect(self.shape)
        for k in range(4):
            shape_tuple = tuple(map(tuple, current_shape))
            variations.add(shape_tuple)
            current_shape = self.rotate_90(current_shape, 1)

        return list(variations)

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self.region_values = board.value_regions()
        # Cache para acelerar verificações
        self._adjacency_cache = {}
        self._empty_regions_cache = None

    def __lt__(self, other):
        return self.id < other.id
    
    def get_empty_regions(self):
        if self._empty_regions_cache is None:
            self._empty_regions_cache = [region for region, value in self.region_values.items() if value == 0]
        return self._empty_regions_cache

class Cell:
    def __init__(self, row, col, region):
        self.row = row
        self.col = col
        self.region = region
        self.blocked_region = None
        self.piece = None

    def value(self):
        return self.piece if self.piece is not None else self.region

class Board:
    def __init__(self, cells):
        self.cells = cells
        self.rows = len(cells)
        self.columns = len(cells[0]) if cells else 0
        print(f"Board initialized with {self.rows} rows and {self.columns} columns")
        self.region_values = {}
        
        # Caches para otimização
        self._region_cells_cache = {}
        self._adjacent_regions_cache = {}
        self._region_sizes_cache = {}
        
        # Pré-computar informações das regiões
        self._precompute_region_info()

    def _precompute_region_info(self):
        """Pré-computa informações sobre regiões para otimização"""
        regions = set()
        for row in self.cells:
            for cell in row:
                if cell.region is not None:
                    regions.add(cell.region)
        
        # Pré-computar células de cada região
        for region in regions:
            self._compute_region_cells(region)

    def _compute_region_cells(self, region):
        """Computa as células de uma região uma vez só"""
        if region in self._region_cells_cache:
            return self._region_cells_cache[region]
            
        cells = []
        for row in range(self.rows):
            for col in range(self.columns):
                if self.cells[row][col].region == region:
                    cells.append(self.cells[row][col])
        
        self._region_cells_cache[region] = cells
        self._region_sizes_cache[region] = len(cells)
        return cells

    def region_cells(self, region):
        return self._compute_region_cells(region)
    
    def region_size(self, region):
        if region not in self._region_sizes_cache:
            self._compute_region_cells(region)
        return self._region_sizes_cache.get(region, 0)

    def adjacent_regions(self, region: int) -> list:
        if region in self._adjacent_regions_cache:
            return self._adjacent_regions_cache[region]
            
        cells = self.region_cells(region)
        neighbours = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for cell in cells:
            for dr, dc in directions:
                r, c = cell.row + dr, cell.col + dc
                if 0 <= r < self.rows and 0 <= c < self.columns:
                    neighbor = self.cells[r][c]
                    if neighbor.region != region and neighbor.region is not None:
                        neighbours.add(neighbor.region)

        result = list(neighbours)
        self._adjacent_regions_cache[region] = result
        return result

    def adjacent_values_cell(self, row: int, col: int) -> list:
        if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
            return []
        
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbours = []
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.columns:
                value = self.cells[r][c].value()
                if value != 'X' and value not in neighbours:
                    neighbours.append(value)
        return neighbours

    def get_value(self, row, column):
        if row < 0 or row >= self.rows or column < 0 or column >= self.columns:
            return None
        return self.cells[row][column].value()

    def get_region(self, row, column):
        if row < 0 or row >= self.rows or column < 0 or column >= self.columns:
            return None
        return self.cells[row][column].region

    def number_of_regions(self):
        regions = set()
        for row in self.cells:
            for cell in row:
                if cell.region is not None:
                    regions.add(cell.region)
        return len(regions)

    @staticmethod
    def get_anchor(variation):
        for i, row in enumerate(variation):
            for j, val in enumerate(row):
                if val == '1':
                    return i, j
        return 0, 0

    def can_place_specific(self, variation, start_row, start_col, piece_value):
        region = self.cells[start_row][start_col].region
        if region is None or self.region_values.get(region, 0) != 0:
            return False

        anchor_row, anchor_col = self.get_anchor(variation)
        piece_positions = []
        
        # Primeira passagem: verificar se pode colocar
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                if value == '1':
                    row = start_row + i - anchor_row
                    col = start_col + j - anchor_col

                    if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
                        return False
                    
                    cell = self.cells[row][col]
                    if cell.piece is not None or cell.region != region:
                        return False
                    
                    piece_positions.append((row, col))

        # Verificar adjacências uma vez só
        has_adjacent_piece = False
        adjacent_region_values = self.adjacent_regions(region)
        
        # Verificar se região vizinha tem peças
        for adj_region in adjacent_region_values:
            if self.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']:
                has_adjacent_piece = True
                break

        # Se há peças vizinhas na região, verificar se toca
        if has_adjacent_piece:
            print("Tenho adjacentes")
            touches_piece = False
            for row, col in piece_positions:
                adjacent_values = self.adjacent_values_cell(row, col)
                for val in adjacent_values:
                    if val in ['L', 'I', 'T', 'S'] and val != piece_value:
                        print("E não sou eu")
                        touches_piece = True
                        break
                if touches_piece:
                    break
            
            if not touches_piece:
                return False

        return True

    def can_place_piece(self, piece, start_row, start_col):
        for variation in piece.variations:
            if self.can_place_specific(variation, start_row, start_col, piece.id):
                return variation
        return None

    def place_specific(self, variation, start_row, start_col, piece_value):
        cell_region = self.cells[start_row][start_col].region
        print(f"Placing piece {piece_value} at ({start_row}, {start_col}) in region {cell_region}")
        self.region_values[cell_region] = piece_value
        anchor_row, anchor_col = self.get_anchor(variation)
        
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                if value == '1' or value == 'X':
                    row = start_row + i - anchor_row
                    col = start_col + j - anchor_col
                    
                    if 0 <= row < self.rows and 0 <= col < self.columns:
                        if value == '1':
                            self.cells[row][col].piece = piece_value
                        elif value == 'X':
                            self.cells[row][col].piece = 'X'
                            self.cells[row][col].blocked_region = (
                            self.cells[row][col].region
                            if self.cells[row][col].region is not None
                            else self.cells[row][col].blocked_region
                            ) 
                            self.cells[row][col].region = None

    def are_pieces_connected(self):
        piece_cells = []
        for row in range(self.rows):
            for col in range(self.columns):
                if self.cells[row][col].piece in ['L', 'I', 'T', 'S']:
                    piece_cells.append((row, col))

        if not piece_cells:
            return False

        # BFS otimizada
        visited = {piece_cells[0]}
        queue = deque([piece_cells[0]])

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.columns and 
                    (nr, nc) not in visited and 
                    self.cells[nr][nc].piece in ['L', 'I', 'T', 'S']):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return len(visited) == len(piece_cells)

    def has_2x2_piece_block(self):
        """Verifica rapidamente se existe um bloco 2x2 de peças"""
        piece_values = {'L', 'I', 'T', 'S'}
        
        for i in range(self.rows - 1):
            for j in range(self.columns - 1):
                square = [
                    self.get_value(i, j),
                    self.get_value(i, j + 1),
                    self.get_value(i + 1, j),
                    self.get_value(i + 1, j + 1)
                ]
                if all(val in piece_values for val in square):
                    return True
        return False

    @staticmethod
    def parse_instance():
        lines = stdin.read().strip().splitlines()
        matrix = [[int(value) for value in line.split()] for line in lines]
        cells = [[Cell(row, column, matrix[row][column]) for column in
                 range(len(matrix[row]))] for row in range(len(matrix))]
        return Board(cells)

    def value_regions(self):
        region_values = {}
        for region in range(1, self.number_of_regions() + 1):
            piece_found = None
            region_cells = self.region_cells(region)
            for cell in region_cells:
                if cell.piece is not None:
                    print(f"Found piece {cell.piece} in region {region} at ({cell.row}, {cell.col})")
                    piece_found = cell.piece
                    break
            print("Region:", region, "Piece found:", piece_found)
            region_values[region] = piece_found if (piece_found is not None and piece_found != 'X') else 0
        return region_values

    def copy(self):
        new_cells = [[Cell(cell.row, cell.col, cell.region) for cell in row] for row in self.cells]

        for r in range(self.rows):
            for c in range(self.columns):
                new_cells[r][c].piece = self.cells[r][c].piece
                new_cells[r][c].blocked_region = self.cells[r][c].blocked_region

        new_board = Board(new_cells)
        new_board.region_values = dict(self.region_values)
        return new_board

    def _show_board_end_(self):
        for row in range(self.rows):
            for column in range(self.columns):
                cell = self.cells[row][column]
                if cell.piece == 'X':
                    print(str(cell.blocked_region) + "\t", end="")
                else:
                    print(str(cell.value()) + "\t", end="")
            print("\n", end="")

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.board = board
        self.initial = NuruominoState(board)
        
        # Pré-computar possibilidades para cada região (otimização major)
        self.possibilities = {}
        self._precompute_all_possibilities()
        
        # Ordenar regiões por dificuldade (heurística de ordenação)
        self.region_order = self._compute_region_order()

    def _precompute_all_possibilities(self):
        """Pré-computa todas as possibilidades para cada região"""
        pieces = [Piece(piece_id) for piece_id in ['L', 'I', 'T', 'S']]
        
        for region in range(1, self.board.number_of_regions() + 1):
            possibilities = []
            region_cells = self.board.region_cells(region)
            
            for cell in region_cells:
                for piece in pieces:
                    for variation in piece.variations:
                        if self.board.can_place_specific(variation, cell.row, cell.col, piece.id):
                            possibilities.append((piece.id, variation, (cell.row, cell.col)))
            
            self.possibilities[region] = possibilities

    def _compute_region_order(self):
        """Ordena regiões por dificuldade de preenchimento (menos possibilidades primeiro)"""
        regions = list(range(1, self.board.number_of_regions() + 1))
        return sorted(regions, key=lambda r: len(self.possibilities.get(r, [])))

    def actions(self, state: NuruominoState):
        if state is None:
            return []
        
        empty_regions = state.get_empty_regions()
        if not empty_regions:
            return []
        
        # Escolher a região com menos possibilidades (MRV - Most Constraining Variable)
        region = min(empty_regions, key=lambda r: len(self.possibilities.get(r, [])))
        
        actions = []
        for piece_id, variation, (row, col) in self.possibilities.get(region, []):
            if state.board.can_place_specific(variation, row, col, piece_id):
                actions.append((Piece(piece_id), variation, row, col))
        
        return actions

    def result(self, state: NuruominoState, action):
        piece, variation, row, col = action
        new_board = state.board.copy()

        if new_board.can_place_specific(variation, row, col, piece.id):
            new_board.place_specific(variation, row, col, piece.id)
            
            # Verificação rápida de 2x2
            if new_board.has_2x2_piece_block():
                return None
                    
            successor = NuruominoState(new_board)
            return successor
        
        return None

    def goal_test(self, state: NuruominoState):
        if state is None:
            return False

        # Verificar se todas as regiões estão preenchidas
        current_regions = state.board.value_regions()
        for region, value in current_regions.items():
            if value == 0:
                return False

        # Verificar conectividade das peças
        if not state.board.are_pieces_connected():
            return False
        
        # Verificação final de 2x2
        if state.board.has_2x2_piece_block():
            return False

        return True

    def h(self, node: Node):
        """Heurística melhorada que considera múltiplos fatores"""
        state = node.state
        if state is None:
            return float('inf')
        
        empty_regions = state.get_empty_regions()
        num_empty = len(empty_regions)
        
        if num_empty == 0:
            return 0
        
        # Penalizar regiões com poucas possibilidades (mais difíceis de preencher)
        constraint_penalty = 0
        for region in empty_regions:
            possibilities = len(self.possibilities.get(region, []))
            if possibilities == 0:
                return float('inf')  # Estado impossível
            constraint_penalty += 1.0 / possibilities
        
        return num_empty + 0.1 * constraint_penalty

if __name__ == "__main__":
    board = Board.parse_instance()

    # Pré-processamento: resolver regiões de tamanho 4 deterministicamente
    for region in range(1, board.number_of_regions() + 1):
        if board.region_size(region) == 4:
            pieces = [Piece('L'), Piece('I'), Piece('T'), Piece('S')]
            region_cells = board.region_cells(region)
            for cell in region_cells:
                placed = False
                for piece in pieces:
                    variation = board.can_place_piece(piece, cell.row, cell.col)
                    if variation is not None:
                        board.place_specific(variation, cell.row, cell.col, piece.id)
                        placed = True
                        break
                if placed:
                    break
    
    board._show_board_end_()
    print("Vamos colocar peça por peça)")
    print("Values:", board.region_values)
    board.place_specific((('1', 'X'), ('1', '1'), ('X', '1')), 0, 2, 'S')
    board._show_board_end_()
    print("Values:", board.region_values)
    #print(f"O 3: {board.region_values[3]}")
    board.place_specific((('1', 'X'), ('1', '1'), ('1', 'X')), 3, 3, 'T')
    board._show_board_end_()
    print("Values:", board.region_values)


    # board.region_values = board.value_regions()
    
    # problem = Nuruomino(board)
    
    # # Usar A* como algoritmo principal (melhor para este tipo de problema)
    # solution = astar_search(problem)
    
    # if solution:
    #     solution.state.board._show_board_end_()
    # else:
    #     print("Nenhuma solução encontrada")