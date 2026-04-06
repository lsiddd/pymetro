# ./systems/ai/fitness.py
import math
from collections import defaultdict
from typing import List, Dict, Any


def _compute_run_length_penalty(line_path: List[int], station_map: Dict, is_loop: bool,
                                 w_adj: float, w_run3: float, w_run4: float) -> float:
    """
    Returns total same-type penalty accounting for run lengths.
    Adjacent pairs: w_adj each.
    Runs of 3: additional w_run3 per triplet.
    Runs of 4+: additional w_run4 per station beyond the triplet.
    """
    path = line_path[:]
    if is_loop and path:
        path = path + [path[0]]  # close the loop for adjacency check

    penalty = 0.0
    n = len(path)
    if n < 2:
        return 0.0

    types = [station_map[sid]['type'] for sid in path]

    i = 0
    while i < n - 1:
        if types[i] == types[i + 1]:
            penalty += w_adj
            # Extend run
            run_len = 2
            j = i + 1
            while j + 1 < n and types[j] == types[j + 1]:
                run_len += 1
                j += 1
            if run_len >= 3:
                penalty += w_run3 * (run_len - 2)
            if run_len >= 4:
                penalty += w_run4 * (run_len - 3)
            i = j
        else:
            i += 1

    return penalty


def calculate_fitness(
    chromosome: Any,
    stations_data: List[Dict],
    seed: int,
    H: int,
    snapshot_week: int = 1,
) -> float:
    """
    Evaluates the chromosome via the real game engine running headless.
    F = w1*ticks - w2*queue_sum - w3*fragility - w4*cost + strategic_bonuses
    """
    from systems.ai.headless_sim import HeadlessSimulation

    sim = HeadlessSimulation(stations_data, chromosome, seed, snapshot_week)
    sim.run(delta_time=50.0, steps=H)

    # --- Base simulation weights ---
    w1 = 10.0
    w2 = 1.0
    w3 = 5.0
    w4 = 50.0
    w5 = 500.0  # Penalty per active line without trains

    # --- Structural weights ---
    w_length = 0.1
    w_excess_stations = 25.0

    # --- Type-alternation weights ---
    w_same_type_adj   = 20.0   # Adjacent same-type pair
    w_same_type_run3  = 40.0   # Extra per station in a run of 3
    w_same_type_run4  = 60.0   # Extra per station in a run of 4+

    # --- Hub weights ---
    w_bad_hub_penalty = 30.0
    w_good_hub_reward = 20.0

    # --- Geometry ---
    w_acute_angle_penalty = 15.0

    # --- Terminal ---
    w_special_terminal_reward = 15.0

    # --- Demand-aware ---
    w_hot_coverage  = 25.0   # Reward per line that reaches the hottest destination type
    w_hot_isolation = 40.0   # Penalty when hottest station has only 1 line connection

    # --- Special coverage ---
    w_special_coverage = 30.0  # Reward × (stations reachable from special / total)

    # --- V-line reward ---
    w_vline_reward = 20.0

    # --- Resource cost ---
    cost = sum(chromosome.trains_per_line) + sum(chromosome.carriages_per_line)
    empty_lines = sum(
        1 for i in range(len(chromosome.lines))
        if len(chromosome.lines[i]) >= 2 and chromosome.trains_per_line[i] == 0
    )

    station_map = {s['id']: s for s in stations_data}
    special_types = {s['type'] for s in stations_data if s['type'] not in ('Circle', 'Triangle', 'Square')}
    total_stations = len(stations_data)

    # --- Hub detection: how many lines pass through each station ---
    station_line_counts: Dict[int, int] = {s['id']: 0 for s in stations_data}
    for line_path in chromosome.lines:
        for s_id in line_path:
            station_line_counts[s_id] += 1

    bad_hub_count = 0
    good_hub_count = 0
    for s_id, count in station_line_counts.items():
        if count > 1:
            stype = station_map[s_id]['type']
            if stype in ('Circle', 'Triangle'):
                bad_hub_count += (count - 1)
            else:
                good_hub_count += (count - 1)

    # --- Demand-aware: which destination type is hottest? ---
    demand_by_type: Dict[str, int] = defaultdict(int)
    for s in stations_data:
        for p_dest in s['passengers']:
            demand_by_type[p_dest] += 1

    hot_types: set = set()
    if demand_by_type:
        max_demand = max(demand_by_type.values())
        threshold = max_demand * 0.75
        hot_types = {t for t, d in demand_by_type.items() if d >= threshold}

    hot_station_ids = {s['id'] for s in stations_data if s['type'] in hot_types}

    hot_coverage_count = 0
    hot_isolation_count = 0
    for line_path in chromosome.lines:
        if any(sid in hot_station_ids for sid in line_path):
            hot_coverage_count += 1

    for hot_sid in hot_station_ids:
        if station_line_counts[hot_sid] < 2:
            hot_isolation_count += 1

    # --- Special station reachability coverage ---
    special_coverage_score = 0.0
    special_station_ids = [s['id'] for s in stations_data if s['type'] in special_types]
    for sp_id in special_station_ids:
        co_stations: set = set()
        for line_path in chromosome.lines:
            if sp_id in line_path:
                co_stations.update(line_path)
        co_stations.discard(sp_id)
        ratio = len(co_stations) / max(1, total_stations - 1)
        special_coverage_score += w_special_coverage * ratio

    # --- Per-line structural metrics ---
    total_physical_length = 0.0
    excess_stations_count = 0
    same_type_penalty = 0.0
    acute_angle_count = 0
    special_terminal_count = 0
    vline_reward = 0.0

    for i in range(len(chromosome.lines)):
        line_path = chromosome.lines[i]
        n_stations = len(line_path)
        if n_stations < 2:
            continue

        is_loop = chromosome.is_loop[i]

        # Terminal special reward (non-loop only)
        if not is_loop:
            for term_id in [line_path[0], line_path[-1]]:
                if station_map[term_id]['type'] not in ('Circle', 'Triangle', 'Square'):
                    special_terminal_count += 1

        # Physical length
        for j in range(n_stations - 1):
            s1, s2 = station_map[line_path[j]], station_map[line_path[j + 1]]
            total_physical_length += math.hypot(s2['x'] - s1['x'], s2['y'] - s1['y'])
        if is_loop:
            s1, s2 = station_map[line_path[-1]], station_map[line_path[0]]
            total_physical_length += math.hypot(s2['x'] - s1['x'], s2['y'] - s1['y'])

        # Run-length aware same-type penalty
        same_type_penalty += _compute_run_length_penalty(
            line_path, station_map, is_loop,
            w_same_type_adj, w_same_type_run3, w_same_type_run4
        )

        # Angle check (penalizes acute angles < 90°)
        triplets = [(line_path[j], line_path[j + 1], line_path[j + 2]) for j in range(n_stations - 2)]
        if is_loop and n_stations >= 3:
            triplets.append((line_path[-2], line_path[-1], line_path[0]))
            triplets.append((line_path[-1], line_path[0], line_path[1]))

        for t_s1, t_s2, t_s3 in triplets:
            p1, p2, p3 = station_map[t_s1], station_map[t_s2], station_map[t_s3]
            vec21_x = p1['x'] - p2['x']
            vec21_y = p1['y'] - p2['y']
            vec23_x = p3['x'] - p2['x']
            vec23_y = p3['y'] - p2['y']
            dot = vec21_x * vec23_x + vec21_y * vec23_y
            if dot > 0.1:
                acute_angle_count += 1

        # Excess stations
        if n_stations > 6:
            excess_stations_count += (n_stations - 6)

        # V-line detection: is the highest-degree station in the middle 50% of the line?
        if not is_loop and n_stations >= 4:
            mid_start = n_stations // 4
            mid_end = 3 * n_stations // 4
            mid_stations = set(line_path[mid_start:mid_end])
            hub_in_mid = any(
                station_line_counts[sid] > 1 and sid in mid_stations
                for sid in line_path
            )
            if hub_in_mid:
                vline_reward += w_vline_reward

    base_fitness = (
        (w1 * sim.ticks_no_overcrowd)
        - (w2 * sim.queue_sum)
        - (w3 * sim.fragile_penalty)
        - (w4 * cost)
        - (w5 * empty_lines)
    )
    spatial_fitness = (
        - (w_length * total_physical_length)
        - (w_excess_stations * excess_stations_count)
    )
    strategic_fitness = (
        (w_special_terminal_reward * special_terminal_count)
        + (w_good_hub_reward * good_hub_count)
        - same_type_penalty
        - (w_bad_hub_penalty * bad_hub_count)
        - (w_acute_angle_penalty * acute_angle_count)
        + (w_hot_coverage * hot_coverage_count)
        - (w_hot_isolation * hot_isolation_count)
        + special_coverage_score
        + vline_reward
    )

    total = base_fitness + spatial_fitness + strategic_fitness

    if sim.game_over:
        survival_ratio = (sim.game_over_tick or 0) / H
        game_over_penalty = 10_000.0 * (1.0 - survival_ratio)
        total -= game_over_penalty

    return total
