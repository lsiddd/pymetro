from collections import deque
import heapq
import itertools # Importado para o contador de desempate

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
    
    # Constrói um grafo de adjacência com informações das linhas
    graph = build_station_graph_with_lines()
    
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
    """Constrói um grafo de adjacência com informações da linha para cada conexão"""
    from state import game_state
    
    graph = {}
    
    # Inicializa o grafo com todas as estações
    for station in game_state.stations:
        graph[station] = {}
    
    # Adiciona conexões das linhas ativas com informações da linha
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

def find_optimal_path(graph, start, end):
    """Encontra o caminho usando o algoritmo A* considerando transferências"""
    if start == end:
        return [start]
    
    if start not in graph or end not in graph:
        return None
    
    # CORREÇÃO: Inicializa um contador único para desempate
    counter = itertools.count() 
    
    # Fila de prioridade: (f_score, g_score, transfers, contador, estação_atual, caminho, ultima_linha)
    open_set = [(0, 0, 0, next(counter), start, [start], None)]
    visited = set()
    
    while open_set:
        # O contador é removido aqui, mas seu valor foi usado para ordenar a fila
        f_score, g_score, transfers, _, current, path, last_line = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == end:
            return path
        
        for neighbor, lines in graph[current].items():
            if neighbor in visited:
                continue
            
            new_g_score = g_score + 1
            new_transfers = transfers
            
            # Verifica se é preciso fazer uma transferência (linha diferente)
            if last_line is not None and not any(line == last_line for line in lines):
                new_transfers += 1
            
            # Escolhe a melhor linha para esta conexão (prefere continuar na mesma linha)
            best_line = None
            if last_line and last_line in lines:
                best_line = last_line
            else:
                best_line = next(iter(lines))  # Pega qualquer linha
            
            # Heurística: distância + penalidade por transferência
            h_score = new_g_score + new_transfers * 2.5
            f_score = h_score
            
            new_path = path + [neighbor]
            # CORREÇÃO: Adiciona o contador à tupla
            heapq.heappush(open_set, (f_score, new_g_score, new_transfers, next(counter), neighbor, new_path, best_line))
    
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