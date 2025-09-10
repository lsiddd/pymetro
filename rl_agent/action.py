# ./rl_agent/action.py
import numpy as np
from state import game_state
from components.train import Train
from rl_agent.state import MAX_STATIONS, MAX_LINES

# Action space definition
# 0: NO_OP
# 1 to (1 + MAX_LINES - 1): add_train_to_line(line_idx)
# (1 + MAX_LINES) to (1 + MAX_LINES*2 - 1): add_carriage_to_line(line_idx)
# (1 + MAX_LINES*2) to (1 + MAX_LINES*3 - 1): remove_line(line_idx)
# ...and so on for line extensions

ACTION_SIZE = 1 + (MAX_LINES * 3) + (MAX_LINES * MAX_STATIONS * 2)

def perform_action(action_index):
    """Executes the action corresponding to the given index."""
    
    if action_index == 0:
        return "NO_OP"

    # --- Resource Actions ---
    if 1 <= action_index < 1 + MAX_LINES:
        line_idx = action_index - 1
        return add_train_to_line(line_idx)
    
    offset = 1 + MAX_LINES
    if offset <= action_index < offset + MAX_LINES:
        line_idx = action_index - offset
        return add_carriage_to_line(line_idx)
    
    offset += MAX_LINES
    if offset <= action_index < offset + MAX_LINES:
        line_idx = action_index - offset
        return remove_line(line_idx)

    # --- Line Extension Actions ---
    offset += MAX_LINES
    if offset <= action_index < offset + (MAX_LINES * MAX_STATIONS):
        flat_idx = action_index - offset
        line_idx = flat_idx // MAX_STATIONS
        station_idx = flat_idx % MAX_STATIONS
        return extend_line(line_idx, station_idx, from_start=True)
    
    offset += MAX_LINES * MAX_STATIONS
    if offset <= action_index < offset + (MAX_LINES * MAX_STATIONS):
        flat_idx = action_index - offset
        line_idx = flat_idx // MAX_STATIONS
        station_idx = flat_idx % MAX_STATIONS
        return extend_line(line_idx, station_idx, from_start=False)

    return "INVALID_ACTION_INDEX"


def get_valid_actions():
    """
    Returns a boolean numpy array of size ACTION_SIZE indicating valid actions.
    This is the core of the action masking logic.
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[0] = True  # NO_OP is always valid

    num_stations = len(game_state.stations)

    # --- Resource Action Validation ---
    for line_idx, line in enumerate(game_state.lines):
        if line_idx >= MAX_LINES: continue

        # Add Train
        if game_state.available_trains > 0 and line.active:
            mask[1 + line_idx] = True
        
        # Add Carriage
        if game_state.carriages > 0 and line.active and any(not train.has_carriage for train in line.trains):
            mask[1 + MAX_LINES + line_idx] = True

        # Remove Line
        if line.active:
            mask[1 + MAX_LINES * 2 + line_idx] = True

    # --- Line Extension Validation ---
    extend_start_offset = 1 + MAX_LINES * 3
    extend_end_offset = extend_start_offset + (MAX_LINES * MAX_STATIONS)
    
    for line_idx in range(game_state.available_lines):
        if line_idx >= MAX_LINES: continue
        line = game_state.lines[line_idx]

        for station_idx in range(num_stations):
            if station_idx >= MAX_STATIONS: continue
            target_station = game_state.stations[station_idx]

            # If station is already on the line, it's an invalid extension target
            if target_station in line.stations:
                continue

            # Case 1: Create a new line (line is not active)
            if not line.active:
                if num_stations >= 2:
                    # To create a line, we need to connect two different stations
                    # Simplified: agent must pick a station to connect to the first station
                    start_station = game_state.stations[0]
                    if target_station != start_station:
                        needs_bridge = line.check_river_crossing(start_station, target_station)
                        if not needs_bridge or game_state.bridges > 0:
                            # Both extend_from_start and extend_from_end are equivalent for a new line
                            flat_idx = line_idx * MAX_STATIONS + station_idx
                            mask[extend_start_offset + flat_idx] = True
            
            # Case 2: Extend an existing line
            else:
                # Extend from start
                start_station = line.stations[0]
                needs_bridge_start = line.check_river_crossing(start_station, target_station)
                if not needs_bridge_start or game_state.bridges > 0:
                    flat_idx = line_idx * MAX_STATIONS + station_idx
                    mask[extend_start_offset + flat_idx] = True

                # Extend from end
                end_station = line.stations[-1]
                needs_bridge_end = line.check_river_crossing(end_station, target_station)
                if not needs_bridge_end or game_state.bridges > 0:
                    flat_idx = line_idx * MAX_STATIONS + station_idx
                    mask[extend_end_offset + flat_idx] = True
    
    return mask


# --- Action Implementation Functions (largely unchanged) ---

def add_train_to_line(line_idx):
    if game_state.available_trains <= 0 or line_idx >= len(game_state.lines): return "INVALID"
    line = game_state.lines[line_idx]
    if not line.active: return "INVALID"
    game_state.available_trains -= 1
    new_train = Train(line)
    game_state.trains.append(new_train)
    line.trains.append(new_train)
    return "SUCCESS_ADD_TRAIN"

def add_carriage_to_line(line_idx):
    if game_state.carriages <= 0 or line_idx >= len(game_state.lines): return "INVALID"
    line = game_state.lines[line_idx]
    if not line.trains: return "INVALID"
    for train in line.trains:
        if not train.has_carriage:
            train.has_carriage = True
            game_state.carriages -= 1
            return "SUCCESS_ADD_CARRIAGE"
    return "INVALID" # All trains on line have carriages

def remove_line(line_idx):
    if line_idx >= len(game_state.lines): return "INVALID"
    line = game_state.lines[line_idx]
    if not line.active: return "INVALID"
    line.clear_line()
    return "SUCCESS_REMOVE_LINE"

def extend_line(line_idx, to_station_idx, from_start):
    if line_idx >= game_state.available_lines or to_station_idx >= len(game_state.stations):
        return "INVALID"
    
    line = game_state.lines[line_idx]
    target_station = game_state.stations[to_station_idx]
    
    if target_station in line.stations: return "INVALID"

    # Case 1: Create a new line
    if not line.active:
        if len(game_state.stations) < 2: return "INVALID"
        # Simplification: A new line always starts from station 0 to the target station
        start_station = game_state.stations[0]
        if target_station == start_station: return "INVALID"
        from_station = start_station
        line.add_station(from_station)
    # Case 2: Extend an existing line
    else:
        from_station = line.stations[0] if from_start else line.stations[-1]

    needs_bridge = line.check_river_crossing(from_station, target_station)
    if needs_bridge and game_state.bridges <= 0: return "FAIL_NO_BRIDGE"

    insert_pos = 0 if from_start else -1
    success = line.add_station(target_station, insert_index=insert_pos)
    
    if success:
        if needs_bridge: game_state.bridges -= 1
        # If a new line was just created, add a train automatically
        if len(line.stations) == 2: add_train_to_line(line_idx)
        return "SUCCESS_EXTEND_LINE"
    else:
        return "INVALID"