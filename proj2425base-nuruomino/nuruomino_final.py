# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 73:
# 109625 Francisco Pestana
# 105953 Gonçalo Simplício
from sys import stdin
import time
from search import *
import numpy as np

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

        # Grafo para verificação da conexão entre peças
        self.region_graph = {}

        # Cache para acelerar verificações
        self._empty_regions_cache = None
        # Cache para conectividade de peças
        self._connectivity_score_cache = {}
        self._empty_adjacent_cache = []


    def __lt__(self, other):
        return self.id < other.id
    
    # Popula o cache das regiões vazias caso ainda não tenha sido feito e retorna as mesmas
    def get_empty_regions(self):
        if self._empty_regions_cache is None:
            self._empty_regions_cache = [region for region, value in self.region_values.items() if value == 0]
        return self._empty_regions_cache
    
    def get_connectivity_score(self, region):
        """Calcula um score de conectividade para uma região baseado em peças adjacentes"""
        if region in self._connectivity_score_cache:
            return self._connectivity_score_cache[region]
        
        adjacent_regions = self.board.adjacent_regions(region)
        connectivity_score = 0
        
        for adj_region in adjacent_regions:
            if self.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']:
                connectivity_score += 1
        
        self._connectivity_score_cache[region] = connectivity_score
        return connectivity_score
    
    #Utilizada para vericar se todas as peças estão conectadas
    def islands_in_graph(self):
        return not self.is_graph_connected()
    
    def is_graph_connected(self):
        if not self.region_graph:
            return True
        
        # Começar de qualquer região
        start_region = next(iter(self.region_graph.keys()))
        visited = {start_region}
        queue = deque([start_region])
        
        while queue:
            current = queue.popleft()
            #print("Current region:", current)
            for neighbor in self.region_graph.get(current, set()):
                #print("Neighbor region:", neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Se visitamos todas as regiões, o grafo está conectado
        #print("region graph:", board.region_graph)
        #print("len visited:", len(visited))
        #print("len region graph:", len(self.region_graph))
        return len(visited) == len(self.region_graph)
    
    def _initialize_region_graph(self):
        graph = {region: set() for region in range(1, self.board.number_of_regions() + 1)}
        empty_regions = self.board.get_empty_regions()
        filled_regions = self.board.get_filled_regions()
        
        # Constrói as conexões entre todas as regiões
        for region in graph:
            neighbours = self.board.adjacent_region_graph(region)
            for neighbour in neighbours:
                if region < neighbour:  # Evitar o processamento duplicado
                    self._connect_regions_if_valid(graph, region, neighbour, empty_regions, filled_regions)
        
        self.region_graph = graph

    # Se é valido conecta duas regiões
    def _connect_regions_if_valid(self, graph, region1, region2, empty_regions, filled_regions):
        # Duas empty, liga tudo
        if region1 in empty_regions and region2 in empty_regions:
            graph[region1].add(region2)
            graph[region2].add(region1)
            return True
        
        # Vê se as peças se tocam
        elif region1 in filled_regions and region2 in filled_regions:
            if self._pieces_touch(region1, region2):
                graph[region1].add(region2)
                graph[region2].add(region1)
                return True
        
        #Uma filled, uma empty
        elif region1 in filled_regions and region2 in empty_regions:
            if self._piece_touches_empty_region(region1, region2):
                graph[region1].add(region2)
                graph[region2].add(region1)
                return True
        #Esquerda filled, direita empty
        elif region1 in empty_regions and region2 in filled_regions:
            if self._piece_touches_empty_region(region2, region1):
                graph[region1].add(region2)
                graph[region2].add(region1)
                return True
        
        return False
    # Verifica se duas peças se tocam
    def _pieces_touch(self, region1, region2):
        region1_coords = self.board.region_piece_coord(region1)
        region2_coords = self.board.region_piece_coord(region2)
        
        for r1, c1 in region1_coords:
            adjacent_coords = self.board.adjacent_coord_cell(r1, c1)
            if any((r, c) in region2_coords for r, c in adjacent_coords):
                return True
        return False
    
    # Verifica se uma peça numa região toca regiões vazias
    def _piece_touches_empty_region(self, filled_region, empty_region):
        piece_coords = self.board.region_piece_coord(filled_region)
        for row, col in piece_coords:
            adjacent_regions = self.board.adjacent_regions_cell(row, col)
            if empty_region in adjacent_regions:
                return True
        return False
    
    #Atualiza o grafo depois da realização de uma ação
    def update_region_graph_incremental(self, placed_region):
        empty_regions = self.board.get_empty_regions()
        filled_regions = self.board.get_filled_regions()
        
        #Só na ortogonalidade
        adjacent_regions = self.board.adjacent_region_graph(placed_region)
        
        #Faz uma cópia para um valor temporário
        old_connections = self.region_graph[placed_region].copy()
        self.region_graph[placed_region].clear()
        
        #Vai tudo embora na bidirecionalidade
        for old_neighbor in old_connections:
            self.region_graph[old_neighbor].discard(placed_region)
        
        # Nas que vamos a mexer
        regions_to_update = [placed_region] + adjacent_regions
        
        #Voltamos a ligar o necessário
        for region in regions_to_update:
            if region == placed_region:
                #Aqui "region" é a placed_region
                for neighbor in adjacent_regions:
                    if self._connect_regions_if_valid_simple(region, neighbor, empty_regions, filled_regions):
                        self.region_graph[region].add(neighbor)
                        self.region_graph[neighbor].add(region)
            else:

                if region in adjacent_regions:
                    if self._connect_regions_if_valid_simple(placed_region, region, empty_regions, filled_regions):
                        self.region_graph[placed_region].add(region)
                        self.region_graph[region].add(placed_region)

    #Só usamos para a verificação de casos
    def _connect_regions_if_valid_simple(self, region1, region2, empty_regions, filled_regions):
        # Duas vazias, liga tudo
        if region1 in empty_regions and region2 in empty_regions:
            return True
        
        #Peças hão de se tocar
        elif region1 in filled_regions and region2 in filled_regions:
            return self._pieces_touch(region1, region2)
        
        #Cheia e vazia
        elif region1 in filled_regions and region2 in empty_regions:
            return self._piece_touches_empty_region(region1, region2)
        
        #Esquerda filled, direita empty
        elif region1 in empty_regions and region2 in filled_regions:
            return self._piece_touches_empty_region(region2, region1)
        
        return False

    # Popula o cache de regiões vazias adjacentes caso ainda não o tenha sido
    def get_empty_adjacent_cache(self):
        if not self._empty_adjacent_cache:
            filled_regions = self.board.get_filled_regions()

            adjacents_final = []
            for reg in filled_regions:
                
                adjacents = self.board.adjacent_regions(reg)
                for adj in adjacents:
                    if adj not in filled_regions and adj not in adjacents_final:
                        adjacents_final.append(adj)    
            self._empty_adjacent_cache = adjacents_final
        
        return self._empty_adjacent_cache


class Cell:
    def __init__(self, row, col, region):
        self.row = row
        self.col = col
        self.region = region
        self.blocked_region = None
        self.piece = None

    def value(self):
        #print("MY value: ", self.piece)
        return self.piece if self.piece is not None else self.region

class Board:
    def __init__(self, cells):
        self.cells = cells
        self.rows = len(cells)
        self.columns = len(cells[0]) if cells else 0
        #print(f"Board initialized with {self.rows} rows and {self.columns} columns")
        self.region_values = {}
        
        
        # Caches para otimização
        self._region_cells_cache = {}
        self._adjacent_regions_cache = {}
        self._adjacent_regions_graph_cache = {}
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
    
    #Devolve as regiões adjacentes
    def adjacent_region_graph(self, region: int) -> list:  
        if region in self._adjacent_regions_graph_cache:
            return self._adjacent_regions_graph_cache[region] 
                
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
        self._adjacent_regions_graph_cache[region] = result
        return result
    
    # Devolve os valores adjacentes de uma célula
    def adjacent_values_cell(self, row: int, col: int) -> list:
        if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
            return []
        
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbours = []
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.columns:
                value = self.cells[r][c].value()
                #print("Desnecessario: ", self.cells[r][c].value())
                #print("VALUE no cell:", value)
                if value != 'X':
                    neighbours.append(value)
        return neighbours
    
    def adjacent_regions_cell(self, row: int, col: int) -> list:
        if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
            return []
        
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbours = set()
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.columns:
                region = self.cells[r][c].region
                if region is not None and region != self.cells[row][col].region:
                    neighbours.add(region)
        return list(neighbours)
    
    def adjacent_coord_cell(self, row: int, col: int) -> list:
        if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
            return []
        
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        neighbours = []
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.columns:
                neighbours.append((r, c))
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

    #Verifica se pode ser colocada uma peça com uma específica orientação
    #não dá check da formação de 2x2 pois é feito no results
    def can_place_specific(self, variation, start_row, start_col, piece_value):
        region = self.cells[start_row][start_col].region
        if region is None or self.region_values.get(region, 0) != 0:
            return False

        anchor_row, anchor_col = self.get_anchor(variation)
        piece_positions = []
        
        # Primeira passagem: verificar se pode colocar
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

                elif value == 'X' and (row >= 0 and row < self.rows) and (col >= 0 and col < self.columns): #So consideramos o X se afeta o tabuleiro, caso contrario nem computa
                    cell = self.cells[row][col]
                    if cell.piece is not None and cell.piece != 'X':
                        return False
        
        adjacent_piece_regions = []
        #Não poder uma peça igual adjacente
        for row, col in piece_positions:   
            adjacent_values = self.adjacent_values_cell(row, col)
            adjacent_piece_regions += self.adjacent_regions_cell(row, col)
            for val in adjacent_values:
                if val in ['L', 'I', 'T', 'S'] and val == piece_value:
                    # print(f"Piece value {piece_value} and val: {val}")
                    return False
        if not adjacent_piece_regions:
            return False #A peça nem toca outras regiões
        
        adjacent_regions = self.adjacent_regions(region)
        regions_with_pieces = []
        
        for adj_region in adjacent_regions:
            if self.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']:
                regions_with_pieces.append(adj_region)
        
        # Se há exatamente uma região adjacente com peça, forçar adjacência

        if (len(regions_with_pieces) == len(adjacent_regions) and regions_with_pieces):
            # Deve tocar na peça da região adjacente obrigatoriamente
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
                return False
            
        return True
    
    #Verifica se pode colocar uma peça iterando pelas suas variações
    def can_place_piece(self, piece, start_row, start_col):
        for variation in piece.variations:
            if self.can_place_specific(variation, start_row, start_col, piece.id):
                return variation
        return None

    #Executa a ação de colocar uma peça no tabuleiro
    def place_specific(self, variation, start_row, start_col, piece_value):
        cell_region = self.cells[start_row][start_col].region
        #print(f"Placing piece {piece_value} at ({start_row}, {start_col}) in region {cell_region}")
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

    #Verifica se existe uma ligação ortogonal entre todas as peças do tabuleiro
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

    #Verifica se existe alguma área do tabuleiro com um 2x2 formado
    def has_2x2_piece_block(self):
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
    
    #Quantas celas ocupadas pode esta peça vir a conectar
    #Podia-se contar peças, mas mais vale contar celulas
    def get_connectivity_potential(self, piece_positions):
        connectivity_score = 0
        
        for row, col in piece_positions:
            adjacent_values = self.adjacent_values_cell(row, col)
            for val in adjacent_values:
                #print("VAL:", val)
                if val in ['L', 'I', 'T', 'S']:
                    connectivity_score += 1
        #print(f"{connectivity_score}")
        return connectivity_score
    
    # Obter o score de conectividade para uma dada variação
    def get_variation_potential(self, variation, start_row, start_col):
        cell = self.cells[start_row][start_col]
        anchor_row, anchor_col = self.get_anchor(variation)
        piece_positions = []
        
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                if value == '1':
                    row = cell.row + i - anchor_row
                    col = cell.col + j - anchor_col
                    if 0 <= row < self.rows and 0 <= col < self.columns:
                        piece_positions.append((row, col))
        
        #print("Piece positions:", piece_positions)
        connectivity_score = self.get_connectivity_potential(piece_positions)
        #print(f"Connectivity potential for piece at ({start_row}, {start_col}): {connectivity_score}")
        return connectivity_score
        

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
                    #print(f"Found piece {cell.piece} in region {region} at ({cell.row}, {cell.col})")
                    piece_found = cell.piece
                    break
            #print("Region:", region, "Piece found:", piece_found)
            region_values[region] = piece_found if (piece_found is not None) else 0
        return region_values

    #Copia um tabuleiro
    def copy(self):
        new_cells = [[Cell(cell.row, cell.col, cell.region) for cell in row] for row in self.cells]

        for r in range(self.rows):
            for c in range(self.columns):
                new_cells[r][c].piece = self.cells[r][c].piece
                new_cells[r][c].blocked_region = self.cells[r][c].blocked_region

        new_board = Board(new_cells)
        new_board.region_values = dict(self.region_values)
        return new_board
    
    #Devolve as coordenadas da peça numa região
    def region_piece_coord(self, region):
        coords = []
        region_cells = self.region_cells(region)
        for cell in region_cells:
            if cell.piece is not None and cell.piece != 'X':
                coords.append((cell.row, cell.col))
        return coords

    #Print do tabuleiro no final do problema
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
    
    #regiões sem peças de um tabuleiro
    def get_empty_regions(self):
        empty_regions = []
        for region in range(1, self.number_of_regions() + 1):
            if self.region_values.get(region, 0) == 0:
                empty_regions.append(region)
        return empty_regions
    
    #regiões com peças de um tabuleiro
    def get_filled_regions(self):
        filled_regions = []
        for region in range(1, self.number_of_regions() + 1):
            if self.region_values.get(region, 0) not in [0, 'X']:
                filled_regions.append(region)
        return filled_regions

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.board = board
        self.initial = NuruominoState(board)
        
        # Pré-computar possibilidades para cada região (otimização)
        self.possibilities = {}
        self._precompute_all_possibilities()

    def _precompute_all_possibilities(self):
        pieces = [Piece(piece_id) for piece_id in ['T', 'I', 'L', 'S']]

        
        for region in range(1, self.board.number_of_regions() + 1):
            possibilities = []
            region_cells = self.board.region_cells(region)
            
            for cell in region_cells:
                for piece in pieces:
                    for variation in piece.variations:
                        if self.board.can_place_specific(variation, cell.row, cell.col, piece.id):
                            # Calcular posições da peça para o connectivity score
                            anchor_row, anchor_col = self.board.get_anchor(variation)
                            piece_positions = []
                            
                            for i, part in enumerate(variation):
                                for j, value in enumerate(part):
                                    if value == '1':
                                        row = cell.row + i - anchor_row
                                        col = cell.col + j - anchor_col
                                        if 0 <= row < self.board.rows and 0 <= col < self.board.columns:
                                            piece_positions.append((row, col))
                            possibilities.append((piece.id, variation, (cell.row, cell.col)))
            self.possibilities[region] = possibilities

    def actions(self, state: NuruominoState):
        if state is None:
            return []
        #print(f"ID de actions: {state.id}")
        filled_regions = state.board.get_filled_regions()
        #print(f"Filled regions: {filled_regions}")


        if not filled_regions:
            empty_regions = state.get_empty_regions()
        else:
            empty_regions = state.get_empty_adjacent_cache()

        if not empty_regions:
            return []

        def region_priority(region):
            num_possibilities = len(self.possibilities.get(region, []))
            if num_possibilities == 0:
                return (float('inf'), 0)
            
            connectivity_score = state.get_connectivity_score(region)
            #print(f"score de conectividade{connectivity_score}")
            # Priorizar: menos possibilidades, maior conectividade
            return (num_possibilities, -connectivity_score)
        
        #Ordena regiões de acordo com a maior conectividade e menor número de possibilidades
        regions = sorted(empty_regions, key=region_priority)
        #print(f"Regions sorted by priority: {regions}")

        if len(regions) > 1:
            if region_priority(regions[0]) == region_priority(regions[1]):
                #print("Emapte: ", regions[0], "e", regions[1])
                mass_tuple = (0, 0)
                center_mass = 0

                num_pieces = (len(filled_regions) if len(filled_regions) != 0 else 1)
                for region in filled_regions:
                    region_cells = state.board.region_cells(region)
                    for cell in region_cells:
                        if cell.piece is not None and cell.piece != 'X':
                            row, col = cell.row, cell.col
                            mass_tuple = (mass_tuple[0] + row, mass_tuple[1] + col)
                
                center_mass = (mass_tuple[0] / (num_pieces * 4), mass_tuple[1] / (num_pieces * 4))
                #print("Center, mass; ", center_mass)

                coords_1 = (0,0)
                coords_2 = (0,0)
                region_1_cells = state.board.region_cells(regions[0])
                region_2_cells = state.board.region_cells(regions[1])

                for cell in region_1_cells:
                    row, col = cell.row, cell.col
                    coords_1 = (coords_1[0] + row, coords_1[1] + col)

                coords_1 = (coords_1[0]/len(region_1_cells), coords_1[1]/len(region_1_cells))

                dist_1 = abs((coords_1[0] - center_mass[0]) + (coords_1[1] - center_mass[1]))
                result_1 = (regions[0], dist_1)


                for cell in region_2_cells:
                    row, col = cell.row, cell.col
                    coords_2 = (coords_2[0] + row, coords_2[1] + col)

                coords_2 = (coords_2[0]/len(region_2_cells), coords_2[1]/len(region_2_cells))

                dist_2 = abs((coords_2[0] - center_mass[0]) + (coords_2[1] - center_mass[1]))
                result_2 = (regions[1], dist_2)

                #print("REsults: ", result_1, result_2)
                region = min(result_1, result_2, key=lambda x: x[1])[0]  # Escolher a região mais próxima do centro de massa  
            else:
                region = regions[0]

        else:
            region = regions[0]  
        
        actions = []

        connect_scores = []
        region_possibilities = self.possibilities.get(region, [])
        #print(f"Possibilities for region {region}: {region_possibilities}")
        for _, variation, (row, col) in region_possibilities:
                var_score = state.board.get_variation_potential(variation, row, col)
                connect_scores.append(var_score)


        #self.connectivity_scores[region] = connect_scores
        region_connectivity_scores = connect_scores
        # print("O meu ID (actions):", state.id)
        # print(f"scores{connect_scores}")
        #print(f"scores {region_connectivity_scores}")
        
            # Combinar possibilidades com scores e ordenar por conectividade
        possibility_data = list(zip(region_possibilities, region_connectivity_scores))
        possibility_data.sort(key=lambda x: x[1])  # Ordenar por conectividade decrescente
    
        for (piece_id, variation, (row, col)), ___ in possibility_data:
            if state.board.can_place_specific(variation, row, col, piece_id):
                #print(f"Action: {piece_id}, {variation}, {row}, {col}")
                actions.append((Piece(piece_id), variation, row, col))
        
        return actions


    def result(self, state: NuruominoState, action):
        piece, variation, row, col = action
        new_board = state.board.copy()
        #print(f"O meu ID: {state.id}")

        if True:  # Sempre verdadeiro, pois já verificamos em actions
            new_board.place_specific(variation, row, col, piece.id)
            #print(f"Placing piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")
            #new_board._show_board_end_()
            #print(" ")
                
            # Verificação rápida de 2x2
            if new_board.has_2x2_piece_block():
                #print("Criamos um 2x2 se formos colocar")
                return None

            successor = NuruominoState(new_board)
            successor.region_graph = {k: v.copy() for k, v in state.region_graph.items()} #copia o grafo
            placed_region = new_board.get_region(row, col)
            successor.update_region_graph_incremental(placed_region)

            if successor.islands_in_graph():
                #print("Criamos uma ilha se formos colocar")
                #print(successor.board.region_graph)
                return None

            #print(f"And created: {successor.id}")
            #new_board._show_board_end_()
                   
            #print(f"Placing piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")
            return successor 
        
        return None

    def goal_test(self, state: NuruominoState):
        if state is None:
            #print("Action deixou de ser valida")
            return False
        
        if state.id == 0:
            #print("State is initial, not a goal")
            state._initialize_region_graph()
            return False
        
        #print(f"Testing goal for state {state.id}")
        #print(" ")
        # Verificar se todas as regiões estão preenchidas
        current_regions = state.board.value_regions()
        for region, value in current_regions.items():
            if value == 0:
                #print("Ha regiões vazias")
                return False

        # Verificar conectividade das peças
        if not state.board.are_pieces_connected():
            #print("Não estão ligadas")
            return False
        
        print("ID: ", state.id)
        return True

    def h(self, node: Node):
        state = node.state
        if state is None:
            return float('inf')
        
        empty_regions = state.get_empty_regions()
        num_empty = len(empty_regions)
        
        if num_empty == 0:
            return 0
        
        heuristic_value = 0
        
        for region in empty_regions:
            possibilities = len(self.possibilities.get(region, []))
            if possibilities == 0:
                return float('inf')  # Estado impossível
            
            # Penalização por baixo número de possibilidades
            constraint_penalty = 1.0 / possibilities
            
            # Bonus para alta conectividade
            connectivity_score = state.get_connectivity_score(region)
            connectivity_bonus = connectivity_score * 0.1
            
            # Penalização por isolamento (regiões sem peças adjacentes)
            isolation_penalty = 0.2 if connectivity_score == 0 else 0
            
            heuristic_value += (constraint_penalty + isolation_penalty - connectivity_bonus)
        
        return num_empty + heuristic_value

if __name__ == "__main__":
    # import time
    start_time = time.time()
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
    
    problem = Nuruomino(board)
    # end_time = time.time()
    # print("\n")
    # print(f"Test completed in {end_time - start_time:.2f} seconds")

    #solution = breadth_first_graph_search(problem)
    solution = depth_first_graph_search(problem)
    # #solution = depth_first_tree_search(problem)
    #solution = astar_search(problem)
    
    if solution:
        #print("\n")
        solution.state.board._show_board_end_()
        #print("Ultimo state: ", solution.state.id)
        end_time = time.time()
        print("\n")
        print(f"Test completed in {end_time - start_time:.2f} seconds")

    else:
        print("Nenhuma solução encontrada")