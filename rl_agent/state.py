# ./rl_agent/state.py
import numpy as np
from state import game_state
from config import CONFIG, STATION_TYPES

# Define fixed dimensions for the state vector
MAX_STATIONS = 20
MAX_LINES = 7
NUM_STATION_TYPES = len(STATION_TYPES.all_types())

# Per station: x, y, is_overcrowded, is_interchange + one-hot for type + passengers for each type
STATION_FEATURE_SIZE = 2 + 1 + 1 + NUM_STATION_TYPES + NUM_STATION_TYPES
# Per line: active + num_trains + num_carriages + sequence of station indices
LINE_FEATURE_SIZE = 1 + 1 + 1 + MAX_STATIONS
# Global resources: available trains, carriages, bridges, interchanges
RESOURCES_SIZE = 4
# Game info
GAME_INFO_SIZE = 2 # score, week
# Enhanced features from the prompt
ENHANCED_FEATURES_SIZE = 4

STATE_SIZE = (MAX_STATIONS * STATION_FEATURE_SIZE) + \
             (MAX_LINES * LINE_FEATURE_SIZE) + \
             RESOURCES_SIZE + GAME_INFO_SIZE + ENHANCED_FEATURES_SIZE


def get_state_vector():
    """Enhanced state representation with better feature engineering"""

    # --- Station Features ---
    station_features = np.zeros(MAX_STATIONS * STATION_FEATURE_SIZE)
    station_id_map = {station.id: i for i, station in enumerate(game_state.stations)}

    for i, station in enumerate(game_state.stations):
        if i >= MAX_STATIONS: break

        offset = i * STATION_FEATURE_SIZE

        # Position (normalized)
        station_features[offset] = station.x / 1200.0
        station_features[offset + 1] = station.y / 800.0

        # Status
        station_features[offset + 2] = 1.0 if station.overcrowd_start_time else 0.0
        station_features[offset + 3] = 1.0 if station.is_interchange else 0.0

        # Type (one-hot)
        type_idx = STATION_TYPES.all_types().index(station.type)
        station_features[offset + 4 + type_idx] = 1.0

        # Waiting passengers (normalized by max capacity)
        passenger_offset = offset + 4 + NUM_STATION_TYPES
        passenger_counts = {stype: 0 for stype in STATION_TYPES.all_types()}
        for p in station.passengers:
            if p.destination in passenger_counts:
                passenger_counts[p.destination] += 1

        for j, stype in enumerate(STATION_TYPES.all_types()):
            station_features[passenger_offset + j] = passenger_counts[stype] / CONFIG.MAX_PASSENGERS_WITH_INTERCHANGE

    # --- Line Features ---
    line_features = np.zeros(MAX_LINES * LINE_FEATURE_SIZE)
    for i, line in enumerate(game_state.lines):
        if i >= MAX_LINES: break

        offset = i * LINE_FEATURE_SIZE

        if line.active:
            line_features[offset] = 1.0
            # Normalized by config max
            line_features[offset + 1] = len(line.trains) / CONFIG.MAX_TRAINS_PER_LINE

            total_carriages = sum(1 for train in line.trains if train.has_carriage)
            # Assume max carriages equals max trains
            line_features[offset + 2] = total_carriages / CONFIG.MAX_TRAINS_PER_LINE

            # Station sequence (use 0 for padding, idx+1 for valid stations)
            station_seq_offset = offset + 3
            for j, station in enumerate(line.stations):
                if j >= MAX_STATIONS: break
                station_idx = station_id_map.get(station.id, -1)
                line_features[station_seq_offset + j] = (station_idx + 1) / MAX_STATIONS

    # --- Resources (Normalized by a reasonable maximum) ---
    resources = np.array([
        game_state.available_trains / 20.0,
        game_state.carriages / 20.0,
        game_state.bridges / 10.0,
        game_state.interchanges / 10.0
    ])

    # --- Game Info (Normalized by a reasonable maximum) ---
    game_info = np.array([
        game_state.score / 5000.0,
        game_state.week / 52.0 # Assume a year's worth of weeks is a high upper bound
    ])
    
    # Concatenate existing state vector components
    existing_vector = np.concatenate([
        station_features,
        line_features,
        resources,
        game_info
    ])

    # Add these additional features:
    
    # Network connectivity metrics
    connected_stations = sum(1 for station in game_state.stations
                           if any(station in line.stations for line in game_state.lines if line.active))
    connectivity_ratio = connected_stations / max(1, len(game_state.stations))

    # Passenger distribution metrics
    waiting_passengers = sum(len(station.passengers) for station in game_state.stations)
    max_waiting = max([len(station.passengers) for station in game_state.stations] + [0])

    # Line utilization metrics
    line_utilizations = []
    for line in game_state.lines:
        if line.active and len(line.stations) >= 2:
            passengers_on_line = sum(len(train.passengers) for train in line.trains)
            capacity = sum(train.total_capacity for train in line.trains)
            utilization = passengers_on_line / max(1, capacity)
            line_utilizations.append(utilization)
    avg_utilization = sum(line_utilizations) / max(1, len(line_utilizations))

    # Add these to your state vector
    enhanced_features = np.array([
        connectivity_ratio,
        waiting_passengers / 50.0,  # normalized
        max_waiting / 10.0,         # normalized
        avg_utilization
    ])
    
    # Concatenate with existing state vector
    state_vector = np.concatenate([existing_vector, enhanced_features]).astype(np.float32)

    # Clip values to ensure they are within a [-1, 1] or [0, 1] range
    np.clip(state_vector, -1.0, 1.0, out=state_vector)

    return state_vector