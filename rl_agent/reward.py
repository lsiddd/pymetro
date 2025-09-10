# ./rl_agent/reward.py
from state import game_state
import math

# Armazena métricas do estado anterior para calcular deltas
previous_metrics = {
    'score': 0,
    'passengers_delivered': 0,
    'total_waiting': 0,
    'overcrowded_stations': 0,
    'connected_stations': 0, # NOVO: Rastreia o número de estações conectadas
}

def update_previous_metrics():
    """Chame isso no início de cada passo para armazenar o estado atual."""
    previous_metrics['score'] = game_state.score
    previous_metrics['passengers_delivered'] = game_state.passengers_delivered
    
    total_waiting = sum(len(station.passengers) for station in game_state.stations)
    overcrowded_stations = sum(1 for station in game_state.stations if station.overcrowd_start_time is not None)
    
    # Uma estação é "conectada" se pertence a pelo menos uma linha ativa
    connected_stations = sum(1 for station in game_state.stations if any(station in line.stations for line in game_state.lines if line.active))
    
    previous_metrics['total_waiting'] = total_waiting
    previous_metrics['overcrowded_stations'] = overcrowded_stations
    previous_metrics['connected_stations'] = connected_stations


def calculate_reward(action_result):
    """
    Calcula a recompensa com base na mudança do estado do jogo, projetada para ensinar maestria.
    Retorna um dicionário com os componentes da recompensa e o total.
    """
    rewards = {
        'delivery_reward': 0.0,             # Camada 1: Objetivo Principal
        'wait_time_penalty': 0.0,           # Camada 2: Saúde do Sistema
        'overcrowding_penalty': 0.0,        # Camada 2: Saúde do Sistema
        'line_efficiency_reward': 0.0,      # Camada 3: Estratégia
        'station_coverage_reward': 0.0,     # Camada 3: Estratégia
        'action_penalty': 0.0,              # Camada 4: Boas Práticas
        'time_penalty': -0.02,              # Camada 4: Boas Práticas
    }

    # --- Camada 1: O Objetivo Principal (Recompensa Massiva por Evento) ---
    passengers_delivered_delta = game_state.passengers_delivered - previous_metrics['passengers_delivered']
    if passengers_delivered_delta > 0:
        rewards['delivery_reward'] = passengers_delivered_delta * 100.0

    # --- Camada 2: Métricas de Saúde do Sistema (Penalidades Contínuas e Severas) ---
    # Penalidade EXPONENCIAL para passageiros esperando. Isso ensina o agente a odiar gargalos.
    current_total_waiting = sum(len(s.passengers) for s in game_state.stations)
    if current_total_waiting > 0:
        # A penalidade cresce mais rápido à medida que mais pessoas esperam
        rewards['wait_time_penalty'] = -math.pow(current_total_waiting, 1.5) * 0.005

    # Penalidade SEVERA e contínua para estações superlotadas. É uma emergência.
    current_overcrowded = sum(1 for s in game_state.stations if s.overcrowd_start_time)
    if current_overcrowded > 0:
        rewards['overcrowding_penalty'] = -current_overcrowded * 10.0
        
    # --- Camada 3: Incentivos de Comportamento Estratégico (Recompensas Contínuas) ---
    # Recompensa por ter uma rede UTILIZADA. Medida pelo número de passageiros nos trens.
    passengers_on_trains = sum(len(train.passengers) for train in game_state.trains)
    if passengers_on_trains > 0:
        rewards['line_efficiency_reward'] = passengers_on_trains * 0.01

    # Bônus por EXPANDIR a rede para cobrir novas estações.
    # current_connected_stations = sum(1 for station in game_state.stations if any(station in line.stations for line in game_state.lines if line.active))
    # newly_connected_stations = current_connected_stations - previous_metrics['connected_stations']
    # if newly_connected_stations > 0:
    #     rewards['station_coverage_reward'] = newly_connected_stations * 5.0

    # --- Camada 4: Guia de Boas Práticas (Penalidades Sutis) ---
    # Penalidade por tentar ações inválidas
    if "INVALID" in action_result or "FAIL" in action_result:
        rewards['action_penalty'] = -0.5
    
    # A penalidade de tempo já está definida na inicialização do dicionário.

    # Calcula a recompensa total
    rewards['total'] = sum(rewards.values())

    return rewards