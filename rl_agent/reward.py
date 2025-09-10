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
    Improved reward function with stable learning signals and better balance.
    Returns a dictionary with reward components and the total.
    """
    rewards = {
        'delivery_reward': 0.0,             # Primary objective - high reward
        'wait_time_penalty': 0.0,           # System health - moderate penalty  
        'overcrowding_penalty': 0.0,        # Critical issue - high penalty
        'line_efficiency_reward': 0.0,      # Strategic behavior - small reward
        'station_coverage_reward': 0.0,     # Network expansion - medium reward
        'action_penalty': 0.0,              # Invalid actions - small penalty
        'survival_reward': 0.1,             # Staying alive - small baseline
    }

    # --- Primary Objective: Passenger Delivery (High Impact) ---
    passengers_delivered_delta = game_state.passengers_delivered - previous_metrics['passengers_delivered']
    if passengers_delivered_delta > 0:
        # Increased reward for delivery to dominate other components
        rewards['delivery_reward'] = passengers_delivered_delta * 50.0

    # --- System Health: Waiting Time (Moderate, Linear Penalty) ---
    current_total_waiting = sum(len(s.passengers) for s in game_state.stations)
    if current_total_waiting > 0:
        # Linear penalty instead of exponential for stable gradients
        # Capped to prevent overwhelming the delivery reward
        rewards['wait_time_penalty'] = -min(current_total_waiting * 0.5, 20.0)

    # --- Critical Issue: Overcrowding (High Penalty) ---
    current_overcrowded = sum(1 for s in game_state.stations if s.overcrowd_start_time)
    if current_overcrowded > 0:
        # Reduced but still significant penalty
        rewards['overcrowding_penalty'] = -current_overcrowded * 5.0
        
    # --- Strategic Behavior: Line Efficiency (Small Positive) ---
    passengers_on_trains = sum(len(train.passengers) for train in game_state.trains)
    if passengers_on_trains > 0:
        # Slightly increased to encourage using the network
        rewards['line_efficiency_reward'] = passengers_on_trains * 0.1

    # --- Network Expansion: Station Coverage (Medium Reward) ---
    current_connected_stations = sum(1 for station in game_state.stations if any(station in line.stations for line in game_state.lines if line.active))
    newly_connected_stations = current_connected_stations - previous_metrics['connected_stations']
    if newly_connected_stations > 0:
        # Reward for expanding network coverage
        rewards['station_coverage_reward'] = newly_connected_stations * 3.0

    # --- Action Quality: Invalid Actions (Small Penalty) ---
    if "INVALID" in action_result or "FAIL" in action_result:
        rewards['action_penalty'] = -1.0
    
    # --- Survival Bonus: Staying Alive ---
    # Small positive reward for each step to encourage longer episodes
    rewards['survival_reward'] = 0.1

    # Calculate total reward
    rewards['total'] = sum(rewards.values())
    
    # Clip total reward to prevent extreme values
    rewards['total'] = max(min(rewards['total'], 100.0), -50.0)

    return rewards