# ./rl_agent/reward.py
from typing import Dict
from state import game_state
import math

# Stores metrics from the previous state to calculate deltas
previous_metrics: Dict[str, int] = {
    'score': 0,
    'passengers_delivered': 0,
    'total_waiting': 0,
    'overcrowded_stations': 0,
    'connected_stations': 0,
    'days_survived': 0,
}

def update_previous_metrics() -> None:
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


# --- CHANGE START ---
def calculate_reward(action_result: str) -> Dict[str, float]:
    """
    Reshaped reward function to focus on the primary goal: passenger delivery.
    - Massive "jackpot" reward for delivering passengers.
    - Small, constant penalty for every step to encourage efficiency.
    - Penalties for waiting passengers and invalid actions.
    - Removed rewards for proxy goals like connecting stations.
    """
    rewards = {
        'delivery_reward': 0.0,
        'time_penalty': 0.0,
        'waiting_penalty': 0.0,
        'action_penalty': 0.0,
    }

    # 1. Jackpot Reward: Large, clear reward for the main objective.
    passengers_delivered_delta = game_state.passengers_delivered - previous_metrics['passengers_delivered']
    if passengers_delivered_delta > 0:
        # Each passenger delivered is a huge win.
        rewards['delivery_reward'] = passengers_delivered_delta * 50.0

    # 2. Time Penalty: A small cost for every step taken.
    # This encourages the agent to achieve its goals efficiently.
    rewards['time_penalty'] = -0.01

    # 3. Waiting Penalty: Punish the agent for letting passengers wait.
    # This encourages creating efficient routes.
    current_waiting = sum(len(s.passengers) for s in game_state.stations)
    previous_waiting = previous_metrics['total_waiting']
    if current_waiting > previous_waiting:
        rewards['waiting_penalty'] = -(current_waiting - previous_waiting) * 0.2

    # 4. Action Penalty: Punish invalid actions to help the agent learn the rules.
    if "INVALID" in action_result or "FAIL" in action_result:
        rewards['action_penalty'] = -0.5

    # Calculate total reward
    total_reward = (rewards['delivery_reward'] +
                   rewards['time_penalty'] +
                   rewards['waiting_penalty'] +
                   rewards['action_penalty'])
    
    return {'total': total_reward, **rewards}
# --- CHANGE END ---