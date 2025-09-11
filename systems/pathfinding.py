from collections import deque
import heapq
import itertools # Importado para o contador de desempate

# Global graph manager instance
class GraphManager:
    """Manages station graph caching to avoid rebuilding the graph every time."""
    def __init__(self):
        self._graph = None
        self._is_dirty = True
    
    def mark_dirty(self):
        """Mark the graph as needing to be rebuilt."""
        self._is_dirty = True
    
    def get_graph(self):
        """Get the current graph, rebuilding if necessary."""
        if self._is_dirty or self._graph is None:
            self._graph = self._build_station_graph_with_lines()
            self._is_dirty = False
        return self._graph
    
    def _build_station_graph_with_lines(self):
        """Internal method to build station graph with line information."""
        from state import game_state
        
        graph = {}
        
        # Initialize graph with all stations
        for station in game_state.stations:
            graph[station] = {}
        
        # Add connections from active lines with line information
        for line in game_state.lines:
            if line.active and len(line.stations) >= 2:
                stations = line.stations
                
                is_loop = len(stations) > 2 and stations[0] == stations[-1]
                
                if is_loop:
                    for i in range(len(stations) - 1):
                        current = stations[i]
                        next_station = stations[(i + 1) % (len(stations) - 1)]
                        
                        if current in graph and next_station in graph:
                            if next_station not in graph[current]:
                                graph[current][next_station] = set()
                            if current not in graph[next_station]:
                                graph[next_station][current] = set()
                            graph[current][next_station].add(line)
                            graph[next_station][current].add(line)
                else:
                    for i in range(len(stations) - 1):
                        current = stations[i]
                        next_station = stations[i + 1]
                        
                        if current in graph and next_station in graph:
                            if next_station not in graph[current]:
                                graph[current][next_station] = set()
                            if current not in graph[next_station]:
                                graph[next_station][current] = set()
                            graph[current][next_station].add(line)
                            graph[next_station][current].add(line)
        
        return graph

# Global graph manager instance
_graph_manager = GraphManager()

def find_path(start_station, destination_type):
    """
    Encontra o caminho ideal considerando tanto a distância quanto o número de transferências.
    Retorna uma lista de estações representando o caminho, ou None se não houver caminho.
    """
    from state import game_state
    
    if not start_station:
        return None
    
    # Encontra todas as estações com o tipo de destino
    destination_stations = [s for s in game_state.stations if s.type == destination_type]
    if not destination_stations:
        return None
    
    # Se a estação inicial já for do tipo de destino, retorna um caminho direto
    if start_station.type == destination_type:
        return [start_station]
    
    # Usa o grafo em cache do GraphManager
    graph = _graph_manager.get_graph()
    
    # Encontra o melhor caminho considerando transferências e distância
    best_path = None
    best_score = float('inf')
    
    for dest_station in destination_stations:
        path = find_optimal_path(graph, start_station, dest_station)
        if path:
            # Pontua o caminho com base no comprimento e no número de transferências
            transfers = count_transfers(path, graph)
            # Penaliza transferências mais severamente que a distância
            score = len(path) + transfers * 2.5
            
            if score < best_score:
                best_score = score
                best_path = path
    
    return best_path

def build_station_graph():
    """Constrói um grafo de adjacência a partir das linhas de metrô ativas"""
    from state import game_state
    
    graph = {}
    
    # Inicializa o grafo com todas as estações
    for station in game_state.stations:
        graph[station] = set()
    
    # Adiciona conexões das linhas ativas
    for line in game_state.lines:
        if line.active and len(line.stations) >= 2:
            stations = line.stations
            
            # Verifica se é um loop (primeira e última estação são a mesma)
            is_loop = len(stations) > 2 and stations[0] == stations[-1]
            
            if is_loop:
                # Para loops, conecta cada estação à próxima (excluindo a duplicata no final)
                for i in range(len(stations) - 1):
                    current = stations[i]
                    next_station = stations[(i + 1) % (len(stations) - 1)]
                    
                    if current in graph and next_station in graph:
                        graph[current].add(next_station)
                        graph[next_station].add(current)  # Bidirecional
            else:
                # Para linhas normais, conecta estações adjacentes
                for i in range(len(stations) - 1):
                    current = stations[i]
                    next_station = stations[i + 1]
                    
                    if current in graph and next_station in graph:
                        graph[current].add(next_station)
                        graph[next_station].add(current)  # Bidirecional
    
    return graph

def bfs_shortest_path(graph, start, end):
    """Encontra o caminho mais curto entre duas estações usando BFS"""
    if start == end:
        return [start]
    
    if start not in graph or end not in graph:
        return None
    
    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in graph[current]:
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

def build_station_graph_with_lines():
    """DEPRECATED: Use GraphManager.get_graph() instead for cached results"""
    return _graph_manager.get_graph()

def mark_graph_dirty():
    """Mark the cached graph as dirty so it will be rebuilt on next access"""
    _graph_manager.mark_dirty()

def find_optimal_path(graph, start, end):
    """Encontra o caminho usando o algoritmo A* otimizado com came_from dictionary"""
    if start == end:
        return [start]
    
    if start not in graph or end not in graph:
        return None
    
    counter = itertools.count()
    
    # Fila de prioridade: (f_score, g_score, transfers, counter, current_station, last_line)
    open_set = [(0, 0, 0, next(counter), start, None)]
    visited = set()
    
    # Dicionários para reconstruir o caminho
    came_from = {}
    g_score = {start: 0}
    transfer_count = {start: 0}
    
    while open_set:
        f_score, current_g, current_transfers, _, current, last_line = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == end:
            # Reconstroi o caminho usando came_from
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            return path[::-1]  # Inverte para ter o caminho do início ao fim
        
        for neighbor, lines in graph[current].items():
            if neighbor in visited:
                continue
            
            new_g_score = current_g + 1
            new_transfers = current_transfers
            
            # Verifica se é preciso fazer uma transferência (linha diferente)
            if last_line is not None and not any(line == last_line for line in lines):
                new_transfers += 1
            
            # Se encontramos um caminho melhor para este vizinho
            if neighbor not in g_score or new_g_score < g_score[neighbor] or \
               (new_g_score == g_score[neighbor] and new_transfers < transfer_count[neighbor]):
                
                came_from[neighbor] = current
                g_score[neighbor] = new_g_score
                transfer_count[neighbor] = new_transfers
                
                # Escolhe a melhor linha para esta conexão (prefere continuar na mesma linha)
                best_line = None
                if last_line and last_line in lines:
                    best_line = last_line
                else:
                    best_line = next(iter(lines))  # Pega qualquer linha
                
                # Heurística: distância + penalidade por transferência
                h_score = new_g_score + new_transfers * 2.5
                f_score = h_score
                
                heapq.heappush(open_set, (f_score, new_g_score, new_transfers, next(counter), neighbor, best_line))
    
    return None

def count_transfers(path, graph):
    """Conta o número de transferências necessárias para um caminho"""
    if len(path) <= 2:
        return 0
    
    transfers = 0
    current_line = None
    
    for i in range(len(path) - 1):
        current_station = path[i]
        next_station = path[i + 1]
        
        if next_station in graph[current_station]:
            available_lines = graph[current_station][next_station]
            
            if current_line is None:
                # Primeiro segmento, escolhe qualquer linha
                current_line = next(iter(available_lines))
            elif current_line not in available_lines:
                # Precisa transferir
                transfers += 1
                current_line = next(iter(available_lines))
    
    return transfers