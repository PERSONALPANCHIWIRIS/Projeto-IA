# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Versão otimizada com estratégia de conectividade e propagação

# Grupo 73:
# 109625 Francisco Pestana
# 105953 Gonçalo Simplício
from sys import stdin
from search import *
import numpy as np
from collections import deque

PIECES = {
    'L': [['0', '1'],
          ['X', '1'],
          ['1', '1']],
    'T': [['1', '1', '1'],
          ['X', '1', 'X']],
    'I': [['1', '1', '1', '1']],
    'S': [['X', '1', '1'],
          ['1', '1', 'X']],
}

class Piece:
    _variations_cache = {}
    
    def __init__(self, id):
        self.id = id
        self.shape = PIECES[id]
        
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
        
        # Cache otimizado
        self._empty_regions_cache = None
        self._filled_regions_cache = None
        self._connectivity_frontier_cache = None

    def __lt__(self, other):
        return self.id < other.id
    
    def get_empty_regions(self):
        if self._empty_regions_cache is None:
            self._empty_regions_cache = [region for region, value in self.region_values.items() if value == 0]
        return self._empty_regions_cache
    
    def get_filled_regions(self):
        if self._filled_regions_cache is None:
            self._filled_regions_cache = [region for region, value in self.region_values.items() if value in ['L', 'I', 'T', 'S']]
        return self._filled_regions_cache
    
    def get_connectivity_frontier(self):
        """Retorna regiões vazias que são adjacentes a regiões com peças"""
        #if self._connectivity_frontier_cache is None:
        filled_regions = set(self.get_filled_regions())
        # for filled in filled_regions:
        #     print(f"Filled region: {filled}")
        empty_regions = set(self.get_empty_regions())
        frontier = set()
        
        for empty_region in empty_regions:
            #print(f"Empty region MAS DENTRO DISTO: {empty_region}")
            adjacent_regions = self.board.adjacent_regions(empty_region)
            #print(f"Adjacent regions for {empty_region}: {adjacent_regions}")
            if any(adj in filled_regions for adj in adjacent_regions):
                frontier.add(empty_region)
        
        self._connectivity_frontier_cache = list(frontier)
        #print(f"Connectivity frontier: {self._connectivity_frontier_cache}")
        return self._connectivity_frontier_cache

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
        
        # Caches para otimização
        self._region_cells_cache = {}
        self._adjacent_regions_cache = {}
        self._region_sizes_cache = {}
        
        # Pré-computar informações das regiões
        self._precompute_region_info()
        self.region_values = self.value_regions()

    def _precompute_region_info(self):
        """Pré-computa informações sobre regiões para otimização"""
        regions = set()
        for row in self.cells:
            for cell in row:
                if cell.region is not None:
                    regions.add(cell.region)
        
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
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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

    def can_place_specific(self, variation, start_row, start_col, piece_value, require_adjacency):
        region = self.cells[start_row][start_col].region
        if region is None or self.region_values.get(region, 0) != 0:
            return False

        anchor_row, anchor_col = self.get_anchor(variation)
        piece_positions = []
        
        # Verificar se pode colocar fisicamente
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                row = start_row + i - anchor_row
                col = start_col + j - anchor_col
                
                if value == '1':
                    if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
                        return False
                    
                    cell = self.cells[row][col]
                    if cell.piece is not None or cell.region != region:
                        return False
                    
                    piece_positions.append((row, col))

                elif value == 'X' and (0 <= row < self.rows) and (0 <= col < self.columns):
                    cell = self.cells[row][col]
                    if cell.piece is not None and cell.piece != 'X':
                        return False
        
        # Verificar adjacências
        adjacent_regions = self.adjacent_regions(region)
        regions_with_pieces = [adj for adj in adjacent_regions if self.region_values.get(adj, 0) in ['L', 'I', 'T', 'S']]
        #print(f"Regions with pieces adjacent to region {region}: {regions_with_pieces}")
        
        # Se não há peças adjacentes e requeremos adjacência, falhar
        if require_adjacency and not regions_with_pieces:
            #print("YOU DONT KNOW IT")
            return False
        
        # Verificar se não há peças iguais adjacentes
        for row, col in piece_positions:
            adjacent_values = self.adjacent_values_cell(row, col)
            for val in adjacent_values:
                if val == piece_value:
                    #print(f"Piece {piece_value} is adjacent to itself at ({row}, {col})")
                    return False
        
        # Se há peças adjacentes, deve tocar pelo menos uma
        if regions_with_pieces:
            touches_adjacent_piece = False
            for row, col in piece_positions:
                adjacent_values = self.adjacent_values_cell(row, col)
                for val in adjacent_values:
                    if val in ['L', 'I', 'T', 'S'] and val != piece_value:
                        touches_adjacent_piece = True
                        break
                if touches_adjacent_piece:
                    break
            
            if not touches_adjacent_piece:
                #print("NEM TOUCA")
                return False
        #print(f"We can place piece {piece_value} at ({start_row}, {start_col}) with variation {variation} in region {region}")
        return True

    def place_specific(self, variation, start_row, start_col, piece_value):
        cell_region = self.cells[start_row][start_col].region
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

        # BFS para verificar conectividade
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
        """Verifica se existe um bloco 2x2 de peças"""
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
                if cell.piece is not None and cell.piece != 'X':
                    piece_found = cell.piece
                    break
            region_values[region] = piece_found if (piece_found is not None) else 0
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
        for row in range(self.rows - 1):
            for column in range(self.columns):
                if column == (self.columns - 1):
                    cell = self.cells[row][column]
                    if cell.piece == 'X':
                        print(str(cell.blocked_region), end="")
                    else:
                        print(str(cell.value()), end="")
                else:
                    cell = self.cells[row][column]
                    if cell.piece == 'X':
                        print(str(cell.blocked_region) + "\t", end="")
                    else:
                        print(str(cell.value()) + "\t", end="")
            print("\n", end="")

        row += 1
        for column in range(self.columns):
            if column == (self.columns - 1):
                cell = self.cells[row][column]
                if cell.piece == 'X':
                    print(str(cell.blocked_region), end="")
                else:
                    print(str(cell.value()), end="")
            else:
                cell = self.cells[row][column]
                if cell.piece == 'X':
                    print(str(cell.blocked_region) + "\t", end="")
                else:
                    print(str(cell.value()) + "\t", end="")

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.board = board
        self.initial = NuruominoState(board)
        
        # Pré-computar possibilidades otimizadas
        self.possibilities = {}
        self._precompute_all_possibilities()

    def _precompute_all_possibilities(self):
        """Pré-computa possibilidades para cada região"""
        pieces = [Piece(piece_id) for piece_id in ['I', 'T', 'S', 'L']]  # Ordem por complexidade
        
        for region in range(1, self.board.number_of_regions() + 1):
            possibilities = []
            region_cells = self.board.region_cells(region)
            
            for cell in region_cells:
                for piece in pieces:
                    for variation in piece.variations:
                        # Permitir colocação inicial sem adjacência se necessário
                        if self.board.can_place_specific(variation, cell.row, cell.col, piece.id, require_adjacency=False):
                            possibilities.append((piece.id, variation, (cell.row, cell.col)))

            self.possibilities[region] = possibilities

    def _get_next_region_strategic(self, state: NuruominoState):
        """Estratégia otimizada para escolher próxima região"""
        empty_regions = state.get_empty_regions()
        if not empty_regions:
            return None
        
        filled_regions = state.get_filled_regions()
        
        # Se não há peças colocadas, começar pela região com menos possibilidades
        if not filled_regions:
            return min(empty_regions, key=lambda r: len(self.possibilities.get(r, [])))
        
        # Priorizar regiões na fronteira de conectividade
        frontier = state.get_connectivity_frontier()
        if frontier:
            # Entre as da fronteira, escolher a com menos possibilidades válidas
            #print(f"WHAT??? {state.get_connectivity_frontier()}")
        #     valid_possibilities = {}
        #     for region in frontier:
        #         #print("Região de fronteira:", region)
        #         count = 0
        #         for piece_id, variation, (row, col) in self.possibilities.get(region, []):
        #             if state.board.can_place_specific(variation, row, col, piece_id, require_adjacency=True):
        #                 count += 1
        #         valid_possibilities[region] = count
            
        #     # Escolher região com menos possibilidades válidas (mais restritiva)
        #     return min(
        #         (r for r in frontier if valid_possibilities.get(r, 0) > 0),
        #         key=lambda r: valid_possibilities.get(r, float('inf')),
        #         default=None
        #     )
        
        # # Se não há fronteira, escolher qualquer região vazia com menos possibilidades
        # return min((r for r in empty_regions if len(self.possibilities.get(r, [])) > 0),
        # key=lambda r: len(self.possibilities.get(r, [])),
        # default=None)
            return frontier

    def actions(self, state: NuruominoState):
        
        if state is None:
            return []
        
        #print("ID de actions:", state.id)
        
        empty_regions = state.get_empty_regions()
        # for empty_region in empty_regions:
        #     #print(f"Region thats empty: {empty_region}")
        if not empty_regions:
            return []
        
        actions = []
        filled_regions = state.get_filled_regions()
        require_adjacency = len(filled_regions) > 0  # Só primeira peça pode ser colocada sem adjacência

        frontier = self._get_next_region_strategic(state)
        if isinstance(frontier, int):
            for piece_id, variation, (row, col) in self.possibilities.get(frontier, []):
                #print(f"Checking piece {piece_id} with variation {variation} at ({row}, {col})")
                #print("VAMOS AVALIAR")
                if state.board.can_place_specific(variation, row, col, piece_id, require_adjacency=require_adjacency):
                    actions.append((Piece(piece_id), variation, row, col))
        else:
            for region in frontier:
                #print(f"Next region to consider: {region}")
                for piece_id, variation, (row, col) in self.possibilities.get(region, []):
                    #print(f"Checking piece {piece_id} with variation {variation} at ({row}, {col})")
                    #print("VAMOS AVALIAR")
                    if state.board.can_place_specific(variation, row, col, piece_id, require_adjacency=require_adjacency):
                        actions.append((Piece(piece_id), variation, row, col))
                        #print(f"Possible action: Place piece {piece_id} at ({row}, {col}) with variation {variation}, region: {state.board.get_region(row, col)}")
                
                #print("All possible actions for region", region, ":", actions)
        return actions

    def result(self, state: NuruominoState, action):
        piece, variation, row, col = action
        new_board = state.board.copy()
        #print(f"O meu ID: {state.id}")
        #new_board._show_board_end_()
        #print("\n")

        if new_board.can_place_specific(variation, row, col, piece.id, require_adjacency=False):
            new_board.place_specific(variation, row, col, piece.id)
                
            # Verificação rápida de 2x2
            if new_board.has_2x2_piece_block():
                #print("Criamos um 2x2 se formos colocar")
                return None
                    
            #print(f"We placed piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")
            successor = NuruominoState(new_board)
            #print(f"And created: {successor.id}")
            #new_board._show_board_end_()
            #print("\n")
            
            
            return successor
        
        return None

    def goal_test(self, state: NuruominoState):
        
        if state is None:
            return False
        
        #print("ID de goal_test:", state.id)

        # Verificar se todas as regiões estão preenchidas
        current_regions = state.board.value_regions()
        for region, value in current_regions.items():
            if value == 0:
                #print("Irei chegar aqui")
                return False

        # Verificar conectividade das peças
        if not state.board.are_pieces_connected():
            return False
        
        # Verificação final de 2x2
        if state.board.has_2x2_piece_block():
            return False

        return True

    def h(self, node: Node):
        """Heurística otimizada"""
        state = node.state
        if state is None:
            return float('inf')
        
        empty_regions = state.get_empty_regions()
        num_empty = len(empty_regions)
        
        if num_empty == 0:
            return 0
        
        heuristic_value = 0
        filled_regions = set(state.get_filled_regions())
        
        for region in empty_regions:
            # Contar possibilidades válidas considerando adjacência
            valid_possibilities = 0
            require_adjacency = len(filled_regions) > 0
            
            for piece_id, variation, (row, col) in self.possibilities.get(region, []):
                if state.board.can_place_specific(variation, row, col, piece_id, require_adjacency=require_adjacency):
                    valid_possibilities += 1
            
            if valid_possibilities == 0:
                return float('inf')  # Estado impossível
            
            # Penalizar regiões com poucas possibilidades
            constraint_penalty = 1.0 / valid_possibilities
            
            # Bonus para regiões na fronteira de conectividade
            adjacent_regions = state.board.adjacent_regions(region)
            connectivity_bonus = 0.1 * sum(1 for adj in adjacent_regions if adj in filled_regions)
            
            heuristic_value += constraint_penalty - connectivity_bonus
        
        return num_empty + heuristic_value

if __name__ == "__main__":
    import time
    start_time = time.time()
    board = Board.parse_instance()

    # Pré-processamento: resolver regiões de tamanho 4 primeiro
    pieces_for_preprocessing = [Piece('I'), Piece('L'), Piece('T'), Piece('S')]
    for region in range(1, board.number_of_regions() + 1):
        if board.region_size(region) == 4:
            region_cells = board.region_cells(region)
            for cell in region_cells:
                placed = False
                for piece in pieces_for_preprocessing:
                    for variation in piece.variations:
                        if board.can_place_specific(variation, cell.row, cell.col, piece.id, require_adjacency=False):
                            board.place_specific(variation, cell.row, cell.col, piece.id)
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
    
    problem = Nuruomino(board)
    
    # Tentar A* primeiro, depois DFS como fallback
    #solution = astar_search(problem)
    solution = depth_first_graph_search(problem)
    
    # if not solution:
    #     solution = depth_first_graph_search(problem)
    
    if solution:
        solution.state.board._show_board_end_()
        end_time = time.time()
        print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    else:
        print("Nenhuma solução encontrada")