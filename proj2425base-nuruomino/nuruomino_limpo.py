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

        #Desta forma constroi sempre em cada estado
        #self.region_graph = self._initialize_region_graph()

        self.region_graph = {}

        # Cache para acelerar verificações
        self._adjacency_cache = {}
        self._empty_regions_cache = None
        # Cache para conectividade de peças
        self._connectivity_score_cache = {}
        self._empty_adjacent_cache = []

    def __lt__(self, other):
        return self.id < other.id
    
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
        
        # Build connections between all regions
        for region in graph:
            neighbours = self.board.adjacent_region_graph(region)
            for neighbour in neighbours:
                if region < neighbour:  # Avoid duplicate processing
                    self._connect_regions_if_valid(graph, region, neighbour, empty_regions, filled_regions)
        
        self.region_graph = graph
        #return graph
    
    def _connect_regions_if_valid(self, graph, region1, region2, empty_regions, filled_regions):
        # Duas empty, liga tudo
        if region1 in empty_regions and region2 in empty_regions:
            graph[region1].add(region2)
            graph[region2].add(region1)
            return True
        
        # Ve se as peças se tocam
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
    
    def _pieces_touch(self, region1, region2):
        region1_coords = self.board.region_piece_coord(region1)
        region2_coords = self.board.region_piece_coord(region2)
        
        for r1, c1 in region1_coords:
            adjacent_coords = self.board.adjacent_coord_cell(r1, c1)
            if any((r, c) in region2_coords for r, c in adjacent_coords):
                return True
        return False
    
    def _piece_touches_empty_region(self, filled_region, empty_region):
        piece_coords = self.board.region_piece_coord(filled_region)
        for row, col in piece_coords:
            adjacent_regions = self.board.adjacent_regions_cell(row, col)
            if empty_region in adjacent_regions:
                return True
        return False
    
    def update_region_graph_incremental(self, placed_region):
        empty_regions = self.board.get_empty_regions()
        filled_regions = self.board.get_filled_regions()
        
        #Só na ortogonalidade
        adjacent_regions = self.board.adjacent_region_graph(placed_region)
        
        #Faz uma copia para ummvalor temporario
        old_connections = self.region_graph[placed_region].copy()
        self.region_graph[placed_region].clear()
        
        #Vai tudo embora na bidirecionalidade
        for old_neighbor in old_connections:
            self.region_graph[old_neighbor].discard(placed_region)
        
        # Nas que vamos a mexer
        regions_to_update = [placed_region] + adjacent_regions
        
        #Voltamos a ligar o necessario
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

    def get_empty_adjacent_cache(self):
        if not self._empty_adjacent_cache:
            #empty_regions = self.get_empty_regions()
            filled_regions = self.board.get_filled_regions()
            #adjacent_empty = set()

            # for filled_region in filled_regions:
            #     adjacent = self.board.adjacent_regions(filled_region)
            #     for region in adjacent:
            #         if region in empty_regions:
            #             adjacent_empty.add(region)

            #self._empty_adjacent_cache = list(adjacent_empty)



            #print("TRincão joga hoje")
            adjacents_final = []
            for reg in filled_regions:
                #print("Reg: ", reg)
                
                adjacents = self.board.adjacent_regions(reg)
                #print("Adjacents:", adjacents)
                for adj in adjacents:
                    if adj not in filled_regions and adj not in adjacents_final:
                        adjacents_final.append(adj)    
            #print("Adjacents final:", adjacents_final)      
            #empty_regions = adjacents_final
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
        self._region_piece_adjacency_cache = {}
        
        # Pré-computar informações das regiões
        self._precompute_region_info()
        #COLOQUEI EU
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
        #print("MWMWMWMWMWMWMWMWMWMWMWMWMWMW")
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
                    # row = start_row + i - anchor_row
                    # col = start_col + j - anchor_col

                    if row < 0 or row >= self.rows or col < 0 or col >= self.columns:
                        return False
                    
                    cell = self.cells[row][col]
                    if cell.piece is not None or cell.region != region:
                        return False
                    
                    piece_positions.append((row, col))

                elif value == 'X' and (row >= 0 and row < self.rows) and (col >= 0 and col < self.columns): #So consideramos o X se afeta o tabuleiro, caso contrario nem computa
                    # row = start_row + i - anchor_row
                    # col = start_col + j - anchor_col
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
                    #print("Sou eu")
                    return False
        if not adjacent_piece_regions:
            return False #A peça nem toca outras regiões
        
        # # Verificação expandida para detectar blocos 2x2 dentro de uma janela 3x3
        # piece_values = {'L', 'I', 'T', 'S'}
        # for center_row, center_col in piece_positions:
        #     for i in range(center_row - 1, center_row + 2):  # linhas
        #         for j in range(center_col - 1, center_col + 2):  # colunas
        #                 square = [
        #                     self.get_value(i, j),
        #                     self.get_value(i, j + 1),
        #                     self.get_value(i + 1, j),
        #                     self.get_value(i + 1, j + 1)
        #                 ]
        #                 #print("CHECKINGGGGGG", square)
        #                 if all(val in piece_values for val in square):
        #                     return False #Criaria um 2x2
                
        #ESTE CA é O NOVO
        adjacent_regions = self.adjacent_regions(region)
        regions_with_pieces = []
        
        for adj_region in adjacent_regions:
            if self.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']:
                regions_with_pieces.append(adj_region)
        
        # Se há exatamente uma região adjacente com peça, forçar adjacência
        #JUntei com a de baixo, mesma logica, aplica para qualquer dos dois casos

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

    def can_place_piece(self, piece, start_row, start_col):
        for variation in piece.variations:
            if self.can_place_specific(variation, start_row, start_col, piece.id):
                return variation
        return None

    def place_specific(self, variation, start_row, start_col, piece_value):
        cell_region = self.cells[start_row][start_col].region
        #print(f"Placing piece {piece_value} at ({start_row}, {start_col}) in region {cell_region}")
        self.region_values[cell_region] = piece_value
        anchor_row, anchor_col = self.get_anchor(variation)
        #piece_positions = []
        
        for i, part in enumerate(variation):
            for j, value in enumerate(part):
                if value == '1' or value == 'X':
                    row = start_row + i - anchor_row
                    col = start_col + j - anchor_col
                    
                    if 0 <= row < self.rows and 0 <= col < self.columns:
                        # piece_positions.append((row, col))
                        if value == '1':
                            self.cells[row][col].piece = piece_value
                            #piece_positions.append((row, col))
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
    
    #Quantas celas ocupadas pode esta peça vir a conectar
    #Podia-se contar peças, mas acho que mais vale contar celas
    def get_connectivity_potential(self, piece_positions):
        connectivity_score = 0
        
        for row, col in piece_positions:
            adjacent_values = self.adjacent_values_cell(row, col)
            for val in adjacent_values:
                if val in ['L', 'I', 'T', 'S']:
                    connectivity_score += 1
        
        return connectivity_score
    
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
        
        connectivity_score = self.get_connectivity_potential(piece_positions)
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

    def copy(self):
        new_cells = [[Cell(cell.row, cell.col, cell.region) for cell in row] for row in self.cells]

        for r in range(self.rows):
            for c in range(self.columns):
                new_cells[r][c].piece = self.cells[r][c].piece
                new_cells[r][c].blocked_region = self.cells[r][c].blocked_region

        new_board = Board(new_cells)
        new_board.region_values = dict(self.region_values)
        #new_board.region_graph = dict(self.region_graph)  # Copiar o grafo de regiões
        return new_board

    def region_piece_coord(self, region):
        coords = []
        region_cells = self.region_cells(region)
        for cell in region_cells:
            if cell.piece is not None and cell.piece != 'X':
                coords.append((cell.row, cell.col))
        return coords


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
    
    def get_empty_regions(self):
        empty_regions = []
        for region in range(1, self.number_of_regions() + 1):
            if self.region_values.get(region, 0) == 0:
                empty_regions.append(region)
        return empty_regions
    
    def get_filled_regions(self):
        filled_regions = []
        for region in range(1, self.number_of_regions() + 1):
            if self.region_values.get(region, 0) not in [0, 'X']:
                filled_regions.append(region)
        return filled_regions

    def build_region_graph(self): #Ao inicio está tudo ligado com tudo
        graph = {region: set() for region in range(1, self.number_of_regions() + 1)}
        empty_regions = self.get_empty_regions()
        filled_regions = self.get_filled_regions()
        visited = set()
        for region in graph:
            if region in visited:
                pass
            neighbours = self.adjacent_region_graph(region)
            for neighbour in neighbours:
                if neighbour in visited:
                    pass

                if region in empty_regions and neighbour in empty_regions:
                    graph[region].add(neighbour)
                    graph[neighbour].add(region)  #nos dois sentidos

                elif region in filled_regions and neighbour in filled_regions:
                    my_piece_coord = self.region_piece_coord(region)
                    neighbour_piece_coord = self.region_piece_coord(neighbour)
                    pieces_touch = False

                    for piece_row, piece_col in my_piece_coord:
                        adjacent_coords = self.adjacent_coord_cell(piece_row, piece_col)
                        if any((r, c) in neighbour_piece_coord for r, c in adjacent_coords):
                            pieces_touch = True
                            break

                    if pieces_touch:
                        graph[region].add(neighbour)
                        graph[neighbour].add(region)
                        break


                elif region in filled_regions and neighbour in empty_regions:
                    my_piece_coord = self.region_piece_coord(region)
                    for piece_row, piece_col in my_piece_coord:
                        adjacent_regs = self.adjacent_regions_cell(piece_row, piece_col)
                        if neighbour in adjacent_regs:
                            graph[region].add(neighbour)
                            graph[neighbour].add(region)
                            break

                elif region in empty_regions and neighbour in filled_regions:
                    neighbour_piece_coord = self.region_piece_coord(neighbour)
                    for row, col in neighbour_piece_coord:
                        adjacent_regs = self.adjacent_regions_cell(row, col)
                        if region in adjacent_regs:
                            graph[region].add(neighbour)
                            graph[neighbour].add(region)
                            break
                visited.add(neighbour) #Adiciona o vizinho ao conjunto de visitados
            visited.add(region)
                

        #print("Grafo de regiões construído:", graph)
        return graph

    #Depois de colocar uma peça, atualizar o grafo de regiões
    def update_region_graph(self, region):  
        new_adjacent = set()  
        empty_regions = self.get_empty_regions()
        filled_regions = self.get_filled_regions()
        adjacent_to_piece = []
        piece_positions = self.region_piece_coord(region)
        for row, col in piece_positions:
            adjacent_to_cell = self.adjacent_regions_cell(row, col)
            for reg in adjacent_to_cell:
                if reg not in adjacent_to_piece:
                    adjacent_to_piece.append(reg)

        #print("Adjacento to piece:", adjacent_to_piece)
        # print("Empty regions:", empty_regions)
        # print("Filled regions:", filled_regions)
        for adj_region in adjacent_to_piece:
            if adj_region != region: #Evitar contar a própria região

                if adj_region in empty_regions: #estamos a ver a minha região com peça e uma sem peça
                    new_adjacent.add(adj_region)

                elif adj_region in filled_regions: #As duas regiões têm peça
                    #new_adjacent.add(adj_region)

                    neighbour_piece_coord = self.region_piece_coord(adj_region)
                    pieces_touch = False

                    for piece_row, piece_col in piece_positions:
                        adjacent_coords = self.adjacent_coord_cell(piece_row, piece_col)
                        if any((r, c) in neighbour_piece_coord for r, c in adjacent_coords):
                            pieces_touch = True
                            break

                    if pieces_touch:
                        #print("ANTÂO")
                        new_adjacent.add(adj_region)

        # Atualizar o grafo de regiões
        self.region_graph[region] = new_adjacent

        #Manter a bi-direção
        for adj in new_adjacent:
            self.region_graph[adj].add(region)

        #remover na bidirecionalidade
        for r in self.region_graph.keys():
            if r != region and region in self.region_graph[r] and r not in new_adjacent:
                #print("REMOVING REGION:", region, "FROM", r)
                self.region_graph[r].remove(region)
        # print("PINCHES SAPOS")

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.board = board
        self.initial = NuruominoState(board)
        
        # Pré-computar possibilidades para cada região (otimização major)
        self.possibilities = {}
        self.connectivity_scores = {}
        self._precompute_all_possibilities()

    def _precompute_all_possibilities(self):
        """Pré-computa todas as possibilidades para cada região"""
        #pieces = [Piece(piece_id) for piece_id in ['L', 'I', 'T', 'S']]
        pieces = [Piece(piece_id) for piece_id in ['T', 'I', 'L', 'S']]

        
        for region in range(1, self.board.number_of_regions() + 1):
            possibilities = []
            connectivity_scores = []
            region_cells = self.board.region_cells(region)
            
            for cell in region_cells:
                for piece in pieces:
                    for variation in piece.variations:
                        if self.board.can_place_specific(variation, cell.row, cell.col, piece.id):
                            # Calcular posições da peça para o connectivity sciore
                            anchor_row, anchor_col = self.board.get_anchor(variation)
                            piece_positions = []
                            
                            for i, part in enumerate(variation):
                                for j, value in enumerate(part):
                                    if value == '1':
                                        row = cell.row + i - anchor_row
                                        col = cell.col + j - anchor_col
                                        if 0 <= row < self.board.rows and 0 <= col < self.board.columns:
                                            piece_positions.append((row, col))
                            
                            connectivity_score = self.board.get_connectivity_potential(piece_positions)
                            
                            possibilities.append((piece.id, variation, (cell.row, cell.col)))
                            connectivity_scores.append(connectivity_score)


            self.possibilities[region] = possibilities
            self.connectivity_scores[region] = connectivity_scores

    def actions(self, state: NuruominoState):
        if state is None:
            return []
        #
        #print(f"ID de actions: {state.id}")
        filled_regions = state.board.get_filled_regions()
        if not filled_regions:
            empty_regions = state.get_empty_regions()
        else:
            empty_regions = state.get_empty_adjacent_cache()
            # #print("TRincão joga hoje")
            # adjacents_final = []
            # for reg in filled_regions:
            #     #print("Reg: ", reg)
                
            #     adjacents = state.board.adjacent_regions(reg)
            #     #print("Adjacents:", adjacents)
            #     for adj in adjacents:
            #         if adj not in filled_regions and adj not in adjacents_final:
            #             adjacents_final.append(adj)    
            # #print("Adjacents final:", adjacents_final)      
            # empty_regions = adjacents_final

        #empty_regions = state.get_empty_regions()
        #for empty_region in empty_regions:
            #print(f"Region thats empty: {empty_region}")
        if not empty_regions:
            return []
        

        def region_priority(region):
            num_possibilities = len(self.possibilities.get(region, []))
            if num_possibilities == 0:
                return (float('inf'), 0)
            
            connectivity_score = state.get_connectivity_score(region)
            
            # Priorizar: menos possibilidades, maior conectividade
            return (num_possibilities, -connectivity_score)
        
        #Tirei a menor região
        region = min(empty_regions, key=region_priority)
        #sorted_regions = sorted(empty_regions, key=region_priority)
        #print("SORTED REGIONS:", sorted_regions)
        
        # Escolher a região com menos possibilidades (MRV - Most Constraining Variable)
        #region = min(empty_regions, key=lambda r: len(self.possibilities.get(r, [])))
        #print(f"EASIER REGION: {region}")
        #print(f"Porque tem : {len(self.possibilities.get(region, []))} possibilidades")
        
        actions = []
        # for piece_id, variation, (row, col) in self.possibilities.get(region, []):
        #     print(f"Action: {piece_id}, {variation}, {row}, {col}")
        #     if state.board.can_place_specific(variation, row, col, piece_id):
        #         #
                
        #         print(f"Action: {piece_id}, {variation}, {row}, {col}")
        #         actions.append((Piece(piece_id), variation, row, col))


        connect_scores = []
        region_possibilities = self.possibilities.get(region, [])
        for _, variation, (row, col) in region_possibilities:
                var_score = self.board.get_variation_potential(variation, row, col)
                connect_scores.append(var_score)

        self.connectivity_scores[region] = connect_scores
        region_connectivity_scores = connect_scores


        # for region in sorted_regions:
        #     region_possibilities = self.possibilities.get(region, [])
        #     #region_connectivity_scores = self.connectivity_scores.get(region, [])
        #     #EXPERIMENTAR
        #     connect_scores = []
        #     for _, variation, (row, col) in region_possibilities:
        #         var_score = self.board.get_variation_potential(variation, row, col)
        #         connect_scores.append(var_score)
        #     self.connectivity_scores[region] = connect_scores
        #     region_connectivity_scores = self.connectivity_scores.get(region, [])

        
            # Combinar possibilidades com scores e ordenar por conectividade
        possibility_data = list(zip(region_possibilities, region_connectivity_scores))
        possibility_data.sort(key=lambda x: -x[1])  # Ordenar por conectividade decrescente
    
        for (piece_id, variation, (row, col)), ___ in possibility_data:
            if state.board.can_place_specific(variation, row, col, piece_id):
                #print(f"Action: {piece_id}, {variation}, {row}, {col}")
                actions.append((Piece(piece_id), variation, row, col))
            
        #for piece, variation, row, col in actions:
            #print(f"Placing piece {piece.id} at ({row}, {col}) with variation {variation} region {self.board.get_region(row, col)}")
        return actions


    def result(self, state: NuruominoState, action):
        piece, variation, row, col = action
        new_board = state.board.copy()
        #new_board.region_graph = dict(new_board.region_graph)  # Copiar o grafo de regiões
        #print(f"O meu ID: {state.id}")

        #if new_board.can_place_specific(variation, row, col, piece.id):
        if True:  # Sempre verdadeiro, pois já verificamos em actions
            new_board.place_specific(variation, row, col, piece.id)
            #print(f"Placing piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")
            #new_board._show_board_end_()
            #print(" ")
                
           
            # Verificação rápida de 2x2
            if new_board.has_2x2_piece_block():
                #print("Criamos um 2x2 se formos colocar")
                return None
            
            
            
            #state.build_region_graph()  # Atualizar o grafo de regiões após colocar a peça
            
            # if state.islands_in_graph():
            #     #print(new_board.islands_in_graph())
            #     #print("Criamos uma ilha se formos colocar")
            #     #print(state.board.region_graph)
            #     #print(new_board.region_graph)        
            #     # new_board._show_board_end_()
            #     # print(" ")
            #     return None
                        
            # region = new_board.get_region(row, col)
            # adjacent_regions = new_board.adjacent_regions(region)
            # for adj_region in adjacent_regions:
            #     if all(new_board.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']):
            # print("Empty regions:", new_board.get_empty_regions())
            # print("Filled regions:", new_board.get_filled_regions())
            # print("Grafo de adjacencias:", new_board.region_graph)


            #print(f"We placed piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")

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
        
        # Verificação final de 2x2
        # if state.board.has_2x2_piece_block():
        #     #print("Criou-se um 2x2")
        #     return False

        #print("ID: ", state.id)
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
    # start_time = time.time()
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

    solution = depth_first_graph_search(problem)

    
    if solution:
        #print("\n")
        solution.state.board._show_board_end_()
        #end_time = time.time()
        # print("\n")
        # print(f"Test completed in {end_time - start_time:.2f} seconds")

    else:
        print("Nenhuma solução encontrada")

