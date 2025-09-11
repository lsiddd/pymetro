# ./rl_agent/reward.py
from state import game_state
import math

# Stores metrics from the previous state to calculate deltas
previous_metrics = {
    'score': 0,
    'passengers_delivered': 0,
    'total_waiting': 0,
    'overcrowded_stations': 0,
    'connected_stations': 0,
    'days_survived': 0,
}

def update_previous_metrics():
    """Call this at the start of each step to store the current state."""
    previous_metrics['score'] = game_state.score
    previous_metrics['passengers_delivered'] = game_state.passengers_delivered
    
    total_waiting = sum(len(station.passengers) for station in game_state.stations)
    overcrowded_stations = sum(1 for station in game_state.stations if station.overcrowd_start_time is not None)
    
    # A station is "connected" if it belongs to at least one active line
    connected_stations = sum(1 for station in game_state.stations if any(station in line.stations for line in game_state.lines if line.active))
    
    previous_metrics['total_waiting'] = total_waiting
    previous_metrics['overcrowded_stations'] = overcrowded_stations
    previous_metrics['connected_stations'] = connected_stations
    
    # Calculate days survived
    current_days = (game_state.week - 1) * 7 + game_state.day
    previous_metrics['days_survived'] = current_days


def calculate_reward(action_result):
    """More balanced reward function with progressive scaling"""
    rewards = {
        'delivery_reward': 0.0,
        'efficiency_reward': 0.0,
        'network_quality_reward': 0.0,
        'penalties': 0.0,
        'action_quality': 0.0,
    }

    # Progressive delivery reward - higher reward for more deliveries
    passengers_delivered_delta = game_state.passengers_delivered - previous_metrics['passengers_delivered']
    if passengers_delivered_delta > 0:
        # Scale reward based on difficulty (more reward early game)
        difficulty_factor = max(0.5, 1.0 - (game_state.week / 52.0))
        rewards['delivery_reward'] = passengers_delivered_delta * 20.0 * difficulty_factor

    # Efficiency reward - reward for efficient passenger movement
    current_waiting = sum(len(s.passengers) for s in game_state.stations)
    previous_waiting = previous_metrics['total_waiting']

    if current_waiting < previous_waiting:
        rewards['efficiency_reward'] = (previous_waiting - current_waiting) * 0.2
    elif current_waiting > previous_waiting:
        rewards['penalties'] -= (current_waiting - previous_waiting) * 0.1

    # Network quality reward - reward for well-connected networks
    current_connected = sum(1 for station in game_state.stations
                          if any(station in line.stations for line in game_state.lines if line.active))
    previous_connected = previous_metrics['connected_stations']

    if current_connected > previous_connected:
        rewards['network_quality_reward'] = (current_connected - previous_connected) * 2.0

    # Action quality assessment
    if "SUCCESS" in action_result:
        rewards['action_quality'] = 1.0
    elif "INVALID" in action_result or "FAIL" in action_result:
        rewards['action_quality'] = -0.5

    # Calculate total with careful scaling
    total_reward = (rewards['delivery_reward'] +
                   rewards['efficiency_reward'] +
                   rewards['network_quality_reward'] +
                   rewards['penalties'] +
                   rewards['action_quality'])

    # Dynamic scaling based on game progress
    progress_factor = min(1.0, game_state.week / 20.0)
    total_reward *= (1.0 + progress_factor)  # Increase reward scaling as game progresses
    
    return {'total': total_reward, **rewards}