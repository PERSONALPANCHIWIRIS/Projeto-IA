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
        # Cache para acelerar verificações
        self._adjacency_cache = {}
        self._empty_regions_cache = None
        # Cache para conectividade de peças
        self._connectivity_score_cache = {}

    def __lt__(self, other):
        return self.id < other.id
        # if isinstance(other, NuruominoState):
        #     return self.id < other.id
        # return NotImplemented
    
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
        #NOVO: isto aqui seria o tal grafo de adjacencias
        #self.region_graph = self.build_region_graph()
        self.region_graph = {}

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
                
        #ESTE CA é O NOVO
        adjacent_regions = self.adjacent_regions(region)
        regions_with_pieces = []
        
        for adj_region in adjacent_regions:
            if self.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']:
                regions_with_pieces.append(adj_region)
        
        # Se há exatamente uma região adjacente com peça, forçar adjacência
        #JUntei com a de baixo, mesma logica, aplica para qualquer dos dois casos
        #if len(regions_with_pieces) == 1 or (len(regions_with_pieces) == len(adjacent_regions) and regions_with_pieces):
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
        
        # Se todas as regiões adjacentes têm peças, deve tocar pelo menos uma
        # elif len(regions_with_pieces) == len(adjacent_regions) and regions_with_pieces:
        #     touches_adjacent_piece = False
        #     for row, col in piece_positions:
        #         adjacent_values = self.adjacent_values_cell(row, col)
        #         for val in adjacent_values:
        #             if val in ['L', 'I', 'T', 'S'] and val != piece_value:
        #                 touches_adjacent_piece = True
        #                 break
        #         if touches_adjacent_piece:
        #             break
            
        #     if not touches_adjacent_piece:
        #         return False
            
        #FORCED ADJACENCY ANTERIOR
        # # Verificar adjacências uma vez só
        # forced_adjancency = True
        # # neighbour_count = 0
        # adjacent_regions = self.adjacent_regions(region)
        
        # # Verificar se região vizinha tem peças
        # for adj_region in adjacent_regions:
        #     if self.region_values.get(adj_region, 0) not in ['L', 'I', 'T', 'S']:
        #         forced_adjancency = False

        # if forced_adjancency:
        #     #print("Tenho adjacentes")
        #     all_good = False
        #     for row, col in piece_positions:
        #         adjacent_values = self.adjacent_values_cell(row, col)
        #         if not adjacent_values:
        #             pass
        #         for val in adjacent_values:
        #             if val in ['L', 'I', 'T', 'S'] and val != piece_value:
        #                 # print(f"Piece value {piece_value} and val: {val}")
        #                 #print("E não sou eu")
        #                 all_good = True
        #                 break
        #         if all_good:
        #             break
        #         #print("FOI AQUI")
        #     if not all_good:
        #         return False
            
            #return True
            # if self.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']:
            #     #has_adjacent_piece = True
            #     neighbour_count = neighbour_count + 1
        #has_adjacent_piece = (neighbour_count == len(adjacent_region_values))
                

        # # Se há peças vizinhas na região, verificar se toca
        # if has_adjacent_piece:
        #     #print("Tenho adjacentes")
        #     touches_piece = False
        #     for row, col in piece_positions:
        #         adjacent_values = self.adjacent_values_cell(row, col)
        #         for val in adjacent_values:
        #             if val in ['L', 'I', 'T', 'S'] and val == piece_value:
        #                 # print(f"Piece value {piece_value} and val: {val}")
        #                 #print("Sou eu")
        #                 return False
                    
        #             if val in ['L', 'I', 'T', 'S'] and val != piece_value:
        #                 # print(f"Piece value {piece_value} and val: {val}")
        #                 #print("E não sou eu")
        #                 touches_piece = True
        #                 #break

        #         # if touches_piece:
        #         #     break
            
        #     if not touches_piece:
        #         return False

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

        #self.update_region_graph(cell_region, piece_positions)
        self.update_region_graph(cell_region)

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
        new_board.region_graph = dict(self.region_graph)  # Copiar o grafo de regiões
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
        for region in graph:
            neighbours = self.adjacent_region_graph(region)
            for neighbour in neighbours:
                graph[region].add(neighbour)
                graph[neighbour].add(region)  #nos dois sentidos

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

        # print("Adjacento to piece:", adjacent_to_piece)
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

                    # neighbour_val = self.region_values.get(adj_region, 0)
                    # adj_to_piece = []
                    # for piece_row, piece_col in piece_positions:
                    #     adjacent_coords = self.adjacent_coord_cell(piece_row, piece_col)
                    #     for r, c in adjacent_coords:
                    #         if 0 <= r < self.rows and 0 <= c < self.columns:
                    #             val= self.cells[r][c].region
                    #             if neighbour_region == adj_region and neighbour_region not in adj_to_piece:
                    #                 adj_to_piece.append(neighbour_region)

        # Atualizar o grafo de regiões
        self.region_graph[region] = new_adjacent
        # print("PRE-PENDEJADAS")
        # print(self.region_graph)
        # print("LA REGION: ", region)
        #ATUALIZAR 2
        # for r in self.region_graph[region]:
        #     if r != region and r not in new_adjacent:
        #         self.region_graph[r].remove(region)

        # #Manter a bi-direção
        # for adj in new_adjacent:
        #     self.region_graph[adj].add(region)

        #Manter a bi-direção
        for adj in new_adjacent:
            self.region_graph[adj].add(region)

        #remover na bidirecionalidade
        # for r in self.region_graph.keys():
        #     if r != region and region in self.region_graph[r] and r not in new_adjacent:
        #         #print("REMOVING REGION:", region, "FROM", r)
        #         self.region_graph[r].remove(region)
        #print("PINCHES SAPOS")

        

        #APAGAR TUDO - RECONSTRUIR TUDO
        # for r in self.region_graph:
        #     if r != region and region in self.region_graph[r]:
        #         self.region_graph[r].remove(region)
        
        # # Depois, adicionar as novas conexões
        # for adj in new_adjacent:
        #     if adj in self.region_graph:
        #         self.region_graph[adj].add(region)

    def islands_in_graph(self):
        # all_regions = set(self.region_graph.keys())

        # visited = set()
        # start = next(iter(all_regions))  # Começa de qualquer região
        # queue = deque([start])

        # while queue:
        #     current = queue.popleft()
        #     if current in visited:
        #         continue
        #     visited.add(current)#Já visitamos a propria

        #     for neighbor in self.region_graph.get(current, set()):
        #         if neighbor in all_regions and neighbor not in visited:
        #             queue.append(neighbor)

        # # Se nem todas as regiões ativas foram visitadas, há ilhas
        # return visited != all_regions
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

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.board = board
        self.initial = NuruominoState(board)
        
        # Pré-computar possibilidades para cada região (otimização major)
        self.possibilities = {}
        self.connectivity_scores = {}
        self._precompute_all_possibilities()
        
        # Ordenar regiões por dificuldade e conectividade
        #self.region_order = self._compute_region_order()

    def _precompute_all_possibilities(self):
        """Pré-computa todas as possibilidades para cada região"""
        pieces = [Piece(piece_id) for piece_id in ['L', 'I', 'T', 'S']]
        #pieces = [Piece(piece_id) for piece_id in ['I', 'T', 'S', 'L']] #Começamos com a peças com menos variações e continuamos assim
        
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
                            #possibilities.append((piece.id, variation, (cell.row, cell.col)))


            # for piece_id, variation, (cell.row, cell.col) in possibilities:
            #     print(f"For region: {region} the possibilities are: {piece_id}, {variation}, {(cell.row, cell.col)} ")
            self.possibilities[region] = possibilities
            self.connectivity_scores[region] = connectivity_scores

    # def _compute_region_order(self):
    #     """Ordena regiões por dificuldade de preenchimento (menos possibilidades primeiro)"""
    #     regions = list(range(1, self.board.number_of_regions() + 1))

    #     def region_priority(region):
    #         num_possibilities = len(self.possibilities.get(region, []))
    #         if num_possibilities == 0:
    #             return (float('inf'), 0)  # Região impossível
            
    #         # Média de conectividade das possibilidades
    #         avg_connectivity = 0
    #         if region in self.connectivity_scores and self.connectivity_scores[region]:
    #             avg_connectivity = sum(self.connectivity_scores[region]) / len(self.connectivity_scores[region])
            
    #         # Priorizar: menos possibilidades primeiro, maior conectividade como desempate
    #         return (num_possibilities, - avg_connectivity)
        
    #     return sorted(regions, key=region_priority)
    #     #return sorted(regions, key=lambda r: len(self.possibilities.get(r, [])))

    def actions(self, state: NuruominoState):
        if state is None:
            return []
        #
        #print(f"ID de actions: {state.id}")
        
        empty_regions = state.get_empty_regions()
        # for empty_region in empty_regions:
        #     #print(f"Region thats empty: {empty_region}")
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
        region_possibilities = self.possibilities.get(region, [])
        region_connectivity_scores = self.connectivity_scores.get(region, [])
        
        # Combinar possibilidades com scores e ordenar por conectividade
        possibility_data = list(zip(region_possibilities, region_connectivity_scores))
        possibility_data.sort(key=lambda x: -x[1])  # Ordenar por conectividade decrescente
        
        for (piece_id, variation, (row, col)), ___ in possibility_data:
            if state.board.can_place_specific(variation, row, col, piece_id):
                #print(f"Action: {piece_id}, {variation}, {row}, {col}")
                actions.append((Piece(piece_id), variation, row, col))
        
        return actions


    def result(self, state: NuruominoState, action):
        piece, variation, row, col = action
        new_board = state.board.copy()
        #new_board.region_graph = dict(new_board.region_graph)  # Copiar o grafo de regiões
        #print(f"O meu ID: {state.id}")

        if new_board.can_place_specific(variation, row, col, piece.id):
            new_board.place_specific(variation, row, col, piece.id)
            #print(f"Placing piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")
            #new_board._show_board_end_()
                
            # Verificação rápida de 2x2
            if new_board.has_2x2_piece_block():
                #print("Criamos um 2x2 se formos colocar")
                return None
            
            if new_board.islands_in_graph():
                #print(new_board.islands_in_graph())
                #print("Criamos uma ilha se formos colocar")
                #print(state.board.region_graph)
                #print(new_board.region_graph)        
                # new_board._show_board_end_()
                # print(" ")
                return None
                        
            # region = new_board.get_region(row, col)
            # adjacent_regions = new_board.adjacent_regions(region)
            # for adj_region in adjacent_regions:
            #     if all(new_board.region_values.get(adj_region, 0) in ['L', 'I', 'T', 'S']):
            # print("Empty regions:", new_board.get_empty_regions())
            # print("Filled regions:", new_board.get_filled_regions())
            # print("Grafo de adjacencias:", new_board.region_graph)


            #print(f"We placed piece {piece.id} at ({row}, {col}) with variation {variation} region {new_board.get_region(row, col)}")
            successor = NuruominoState(new_board)
            #print(f"And created: {successor.id}")
            #new_board._show_board_end_()
                   
            return successor
        
        return None

    def goal_test(self, state: NuruominoState):
        if state is None:
            #print("Action deixou de ser valida")
            return False
        #print(f"Testing goal for state {state.id}")
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
        if state.board.has_2x2_piece_block():
            #print("Criou-se um 2x2")
            return False

        print("ID: ", state.id)
        return True

    def h(self, node: Node):
        # """Heurística melhorada que considera múltiplos fatores"""
        # state = node.state
        # if state is None:
        #     return float('inf')
        
        # empty_regions = state.get_empty_regions()
        # num_empty = len(empty_regions)
        
        # if num_empty == 0:
        #     return 0
        
        # # Penalizar regiões com poucas possibilidades (mais difíceis de preencher)
        # constraint_penalty = 0
        # for region in empty_regions:
        #     possibilities = len(self.possibilities.get(region, []))
        #     if possibilities == 0:
        #         return float('inf')  # Estado impossível
        #     constraint_penalty += 1.0 / possibilities
        
        # return num_empty + 0.1 * constraint_penalty

        """Heurística otimizada considerando possibilidades, conectividade e constraintes"""
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
    import time
    start_time = time.time()
    board = Board.parse_instance()
    #O grafo de adjacencias 
    board.region_graph = board.build_region_graph()

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
    
    #board._show_board_end_()
    # print("Vamos colocar peça por peça)")
    # print("Values:", board.region_values)
    # board.place_specific((('1', 'X'), ('1', '1'), ('X', '1')), 0, 2, 'S')
    # board._show_board_end_()
    # print("Values:", board.region_values)
    # #print(f"O 3: {board.region_values[3]}")
    # board.can_place_specific((('1', '0'), ('1', 'X'), ('1', '1')), 3, 3, 'L')
    # #board.place_specific((('1', 'X'), ('1', '1'), ('1', 'X')), 3, 3, 'T')
    # board._show_board_end_()
    # print("Values:", board.region_values)


    #board.region_values = board.value_regions()
    
    problem = Nuruomino(board)
    
    # Usar A* como algoritmo principal (melhor para este tipo de problema)
    #solution = astar_search(problem)
    solution = depth_first_graph_search(problem)
    #solution = breadth_first_tree_search(problem)
    #solution = depth_first_tree_search(problem)
    #solution = breadth_first_graph_search(problem)
    #solution = greedy_search(problem)
    #solution = hill_climbing(problem)
    #solution = astar_search(problem, h=problem.h)
    
    # if solution:
    #     #print("Solução encontrada:")
    #     solution.state.board._show_board_end_()
    # else:
    #     print("Nenhuma solução encontrada")

    # if not solution:
    #     # Fallback para busca em profundidade se A* falhar
    #     #print("A* não encontrou solução, tentando busca em profundidade...")
    #     #solution = astar_search(problem)
    #     solution = astar_search(problem, h=problem.h)
    #     #solution = depth_first_graph_search(problem)
    
    if solution:
        # print("\n")
        # print("Solução encontrada:")
        solution.state.board._show_board_end_()
        end_time = time.time()
        print("\n")
        print(f"Test completed in {end_time - start_time:.2f} seconds")

    else:
        #print("region graph:", board.region_graph)
        print("Nenhuma solução encontrada")



#_______________________________________DEBUGGING TESTE 12
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# board.place_specific((('1', 'X'), ('1', '1'), ('X', '1')), 1, 0, 'S')
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# board.place_specific((('1', 'X'), ('1', '1'), ('1', 'X')), 0, 2, 'T')
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# # board.place_specific((('1', '1', '1', '1'),), 4, 1, 'I')
# # board._show_board_end_()
# # print(" ")
# # print("empty:", board.get_empty_regions())
# # print("filled:", board.get_filled_regions())
# # print("region graph:", board.region_graph)
# # board.place_specific((('1', '0'), ('1', 'X'),('1', '1')), 2, 5, 'L')
# # board._show_board_end_()
# # print(" ")
# # print("empty:", board.get_empty_regions())
# # print("filled:", board.get_filled_regions())
# # print("region graph:", board.region_graph)
# board.place_specific((('X', '1'), ('1', '1'),('1', 'X')), 0, 7, 'S')
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# board.place_specific((('X', '1'), ('1', '1'),('X', '1')), 0, 9, 'T')
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# print(board.is_graph_connected())
# print(board.islands_in_graph())
# # board.place_specific((('0', 'X', '1'), ('1', '1', '1')), 8, 4, 'L')
# # board._show_board_end_()
# # print(" ")
# # print("empty:", board.get_empty_regions())
# # print("filled:", board.get_filled_regions())
# # print("region graph:", board.region_graph)
# # board.place_specific((('1',), ('1',), ('1',), ('1',)), 5, 5, 'I')
# # board._show_board_end_()
# # print(" ")
# # print("empty:", board.get_empty_regions())
# # print("filled:", board.get_filled_regions())
# # print("region graph:", board.region_graph)
# board.place_specific((('1', '1'), ('1', 'X'), ('1', '0')), 7, 0, 'L')
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# # board.place_specific((('X', '1'), ('1', '1'), ('1', 'X')), 6, 3, 'S')
# # board._show_board_end_()
# # print(" ")
# # print("empty:", board.get_empty_regions())
# # print("filled:", board.get_filled_regions())
# # print("region graph:", board.region_graph)
# # board.place_specific((('1', '1', '1'), ('X', '1', 'X')), 6, 6, 'T')
# # board._show_board_end_()
# # print(" ")
# # print("empty:", board.get_empty_regions())
# # print("filled:", board.get_filled_regions())
# # print("region graph:", board.region_graph)
# board.place_specific((('1', '1', '1'), ('0', 'X', '1')), 8, 7, 'L')
# board._show_board_end_()
# print(" ")
# print("empty:", board.get_empty_regions())
# print("filled:", board.get_filled_regions())
# print("region graph:", board.region_graph)
# print(board.is_graph_connected())
# print(board.islands_in_graph())