# ./systems/ai/operators.py
import random
import math
from collections import defaultdict
from typing import List, Any, Dict
from config import STATION_TYPES
from .chromosome import Chromosome

_LOOP_DIRECTIONS = ('both', 'forward', 'backward')


def _alternation_delta(line: List[int], insert_idx: int, new_sid: int, station_map: dict) -> int:
    """
    Returns the change in adjacent same-type pair count when inserting new_sid at insert_idx.
    Negative means fewer same-type pairs (good).
    """
    new_type = station_map[new_sid].type
    delta = 0
    prev_sid = line[insert_idx - 1] if insert_idx > 0 else None
    next_sid = line[insert_idx] if insert_idx < len(line) else None

    # Would the new insertion create same-type adjacency?
    if prev_sid and station_map[prev_sid].type == new_type:
        delta += 1
    if next_sid and station_map[next_sid].type == new_type:
        delta += 1

    # Does the current adjacency (prev→next) that we break save a penalty?
    if prev_sid and next_sid and station_map[prev_sid].type == station_map[next_sid].type:
        delta -= 1

    return delta


def generate_initial_population(N: int, game_state: Any) -> List[Chromosome]:
    population = []

    # Weights for station types
    special_types = STATION_TYPES.special_types()
    weights = {}
    station_map = {s.id: s for s in game_state.stations}

    for station in game_state.stations:
        weight = 10 if station.type in special_types else 1
        weight += len(station.passengers) * 2  # high queue -> high priority
        weights[station.id] = weight

    total_trains = game_state.available_trains + sum(len(line.trains) for line in game_state.lines)
    total_carriages = game_state.carriages

    for _ in range(N):
        c = Chromosome(game_state.available_lines)

        # Populate lines
        for i in range(game_state.available_lines):
            num_stations = random.randint(2, min(8, max(2, len(game_state.stations))))

            # Weighted sampling
            population_pool = []
            pool_weights = []
            for s, w in weights.items():
                if s not in c.lines[i]:
                    population_pool.append(s)
                    pool_weights.append(w)

            if not population_pool:
                continue

            try:
                selected = random.choices(population_pool, weights=pool_weights, k=num_stations)
            except ValueError:
                selected = random.sample(population_pool, min(num_stations, len(population_pool)))

            # Remove duplicates for initial
            selected = list(set(selected))

            # Nearest neighbor ordering using ID mapping
            if selected:
                ordered = [selected.pop(0)]
                while selected:
                    last = ordered[-1]
                    s_last = station_map[last]
                    next_s = min(selected, key=lambda s_id: math.hypot(station_map[s_id].x - s_last.x, station_map[s_id].y - s_last.y))
                    selected.remove(next_s)
                    ordered.append(next_s)
                c.lines[i] = ordered

            if random.random() < 0.4 and len(c.lines[i]) > 2:
                c.is_loop[i] = True
                c.loop_direction[i] = random.choice(_LOOP_DIRECTIONS)

        # Distribute resources randomly
        trains_pool = total_trains
        carriages_pool = total_carriages

        while trains_pool > 0:
            target_line = random.randint(0, game_state.available_lines - 1)
            c.trains_per_line[target_line] += 1
            trains_pool -= 1

        while carriages_pool > 0:
            target_line = random.randint(0, game_state.available_lines - 1)
            c.carriages_per_line[target_line] += 1
            carriages_pool -= 1

        c.repair()
        population.append(c)

    return population


def generate_heuristic_seeds(game_state: Any, n: int = 20) -> List[Chromosome]:
    """
    Builds chromosomes from known-good MiniMetro strategies.
    Returns up to `n` chromosomes mixing four seed strategies.
    """
    stations = game_state.stations
    station_map = {s.id: s for s in stations}
    special_types = set(STATION_TYPES.special_types())
    num_lines = game_state.available_lines

    total_trains = game_state.available_trains + sum(len(line.trains) for line in game_state.lines)
    total_carriages = game_state.carriages

    def _distribute_resources(c: Chromosome, active_lines: List[int]):
        if not active_lines:
            return
        trains_left = total_trains
        carriages_left = total_carriages
        for line_idx in active_lines:
            c.trains_per_line[line_idx] = 1
            trains_left -= 1
            if trains_left <= 0:
                break
        while trains_left > 0:
            target = random.choice(active_lines)
            c.trains_per_line[target] += 1
            trains_left -= 1
        while carriages_left > 0:
            target = random.choice(active_lines)
            c.carriages_per_line[target] += 1
            carriages_left -= 1

    def _nearest_neighbor_order(ids: List[int]) -> List[int]:
        if not ids:
            return []
        ids = list(ids)
        ordered = [ids.pop(0)]
        while ids:
            last = ordered[-1]
            s_last = station_map[last]
            nxt = min(ids, key=lambda sid: math.hypot(station_map[sid].x - s_last.x, station_map[sid].y - s_last.y))
            ids.remove(nxt)
            ordered.append(nxt)
        return ordered

    seeds: List[Chromosome] = []

    # --- Strategy 1: Hub-and-spoke ---
    # Station with highest queue becomes mandatory stop on every line.
    def hub_and_spoke() -> Chromosome:
        c = Chromosome(num_lines)
        hub = max(stations, key=lambda s: len(s.passengers) + (10 if s.type in special_types else 0))
        all_ids = [s.id for s in stations if s.id != hub.id]
        chunk_size = max(2, len(all_ids) // max(1, num_lines))
        random.shuffle(all_ids)
        for i in range(num_lines):
            chunk = all_ids[i * chunk_size:(i + 1) * chunk_size]
            line_ids = [hub.id] + _nearest_neighbor_order(chunk)
            c.lines[i] = line_ids[:8]
        active = [i for i in range(num_lines) if len(c.lines[i]) >= 2]
        _distribute_resources(c, active)
        c.repair()
        return c

    # --- Strategy 2: Alternating-type chains ---
    # Greedily build each line by appending the nearest station of an opposite type.
    def alternating_chains() -> Chromosome:
        c = Chromosome(num_lines)
        all_ids = [s.id for s in stations]
        used: set = set()

        for i in range(num_lines):
            available = [sid for sid in all_ids if sid not in used]
            if len(available) < 2:
                break
            # Start with a random station
            start = random.choice(available)
            chain = [start]
            used.add(start)
            last_type = station_map[start].type

            for _ in range(min(7, len(available) - 1)):
                candidates = [sid for sid in all_ids if sid not in chain]
                if not candidates:
                    break
                # Prefer opposite type, then nearest
                diff_type = [sid for sid in candidates if station_map[sid].type != last_type]
                pool = diff_type if diff_type else candidates
                last_s = station_map[chain[-1]]
                nxt = min(pool, key=lambda sid: math.hypot(station_map[sid].x - last_s.x, station_map[sid].y - last_s.y))
                chain.append(nxt)
                last_type = station_map[nxt].type

            c.lines[i] = chain

        active = [i for i in range(num_lines) if len(c.lines[i]) >= 2]
        _distribute_resources(c, active)
        c.repair()
        return c

    # --- Strategy 3: Special-loop + feeder lines ---
    # All special stations on one big loop; secondary lines feed common types to their nearest special.
    def special_loop() -> Chromosome:
        c = Chromosome(num_lines)
        specials = [s.id for s in stations if s.type in special_types]
        commons = [s.id for s in stations if s.type not in special_types]

        if len(specials) >= 2:
            loop_ids = _nearest_neighbor_order(specials)
            # Sprinkle in common stations to improve alternation
            for sid in _nearest_neighbor_order(commons[:max(0, 8 - len(loop_ids))]):
                loop_ids.append(sid)
            c.lines[0] = loop_ids[:8]
            c.is_loop[0] = True
            c.loop_direction[0] = 'both'
        else:
            # Fallback: longest chain
            c.lines[0] = _nearest_neighbor_order([s.id for s in stations])[:8]

        # Remaining lines: groups of common stations connected to their nearest special
        remaining_commons = [s.id for s in stations if s.id not in c.lines[0]]
        chunk_size = max(2, len(remaining_commons) // max(1, num_lines - 1))
        random.shuffle(remaining_commons)
        for i in range(1, num_lines):
            chunk = remaining_commons[(i - 1) * chunk_size:i * chunk_size]
            if chunk and specials:
                nearest_special = min(specials, key=lambda sp: min(
                    math.hypot(station_map[sp].x - station_map[sid].x, station_map[sp].y - station_map[sid].y)
                    for sid in chunk
                ))
                c.lines[i] = [nearest_special] + _nearest_neighbor_order(chunk)
                c.lines[i] = c.lines[i][:8]

        active = [i for i in range(num_lines) if len(c.lines[i]) >= 2]
        _distribute_resources(c, active)
        c.repair()
        return c

    # --- Strategy 4: Demand-first express ---
    # Identify the hottest destination type, build a dedicated express line to it.
    def demand_first() -> Chromosome:
        c = Chromosome(num_lines)
        demand: Dict[str, int] = defaultdict(int)
        for s in stations:
            for p in s.passengers:
                demand[p] += 1

        hot_type = max(demand, key=demand.get) if demand else None
        hot_stations = [s.id for s in stations if s.type == (hot_type or '')] or [stations[0].id]
        other_stations = [s.id for s in stations if s.id not in hot_stations]

        # Express line: connects source clusters directly to a hot station
        express = hot_stations[:1] + _nearest_neighbor_order(other_stations[:5])
        c.lines[0] = express[:8]

        # Remaining lines distributed normally
        remaining = [s.id for s in stations if s.id not in c.lines[0]]
        chunk_size = max(2, len(remaining) // max(1, num_lines - 1))
        random.shuffle(remaining)
        for i in range(1, num_lines):
            chunk = remaining[(i - 1) * chunk_size:i * chunk_size]
            if chunk:
                c.lines[i] = _nearest_neighbor_order(chunk)[:8]
                if len(c.lines[i]) > 2 and random.random() < 0.3:
                    c.is_loop[i] = True
                    c.loop_direction[i] = 'both'

        active = [i for i in range(num_lines) if len(c.lines[i]) >= 2]
        _distribute_resources(c, active)
        c.repair()
        return c

    strategies = [hub_and_spoke, alternating_chains, special_loop, demand_first]
    per_strategy = max(1, n // len(strategies))

    for strategy_fn in strategies:
        for _ in range(per_strategy):
            try:
                seeds.append(strategy_fn())
            except Exception as exc:
                import logging
                logging.debug("Heuristic seed generation failed (%s): %s", strategy_fn.__name__, exc)

    return seeds[:n]


def tournament_selection(population: List[Chromosome], fitnesses: List[float], k: int = 5) -> Chromosome:
    best_idx = random.randint(0, len(population) - 1)
    best_fit = fitnesses[best_idx]

    for _ in range(k - 1):
        idx = random.randint(0, len(population) - 1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]

    return population[best_idx].copy()


def edge_assembly_crossover(parent_a: Chromosome, parent_b: Chromosome, pc: float = 0.85) -> Chromosome:
    if random.random() > pc:
        return parent_a.copy() if random.random() < 0.5 else parent_b.copy()

    n_lines = len(parent_a.lines)
    child = Chromosome(n_lines)
    child.trains_per_line = parent_a.trains_per_line[:]
    child.carriages_per_line = parent_a.carriages_per_line[:]
    child.is_loop = parent_a.is_loop[:]
    child.loop_direction = parent_a.loop_direction[:]

    # EAX adapted logic
    shared = min(n_lines, len(parent_b.lines))
    for i in range(n_lines):
        line_a = parent_a.lines[i]
        # If parent_b doesn't have this line index, fall back to parent_a's line
        line_b = parent_b.lines[i] if i < shared else []

        if not line_a or not line_b:
            child.lines[i] = line_a[:] if line_a else line_b[:]
            continue

        # Simplified EAX: 2-point crossover on station lists
        if len(line_a) > 2 and len(line_b) > 2:
            pt1 = random.randint(0, len(line_a) - 1)
            pt2 = random.randint(pt1, len(line_a))
            segment = line_a[pt1:pt2]

            new_line = segment[:]
            for s in line_b:
                if s not in new_line:
                    new_line.append(s)

            MAX_STATIONS = 10
            child.lines[i] = new_line[:MAX_STATIONS]
        else:
            child.lines[i] = line_a[:] if random.random() < 0.5 else line_b[:]

    child.repair()
    return child


def _line_length_delta(line: List[int], idx1: int, idx2: int, station_map: dict) -> float:
    """O(1) change in total Euclidean length from reversing line[idx1:idx2] (2-opt delta)."""
    n = len(line)
    if idx1 == 0 or idx2 >= n:
        return 0.0
    a = station_map[line[idx1 - 1]]
    b = station_map[line[idx1]]
    c = station_map[line[idx2 - 1]]
    d = station_map[line[idx2]]
    before = math.hypot(a.x - b.x, a.y - b.y) + math.hypot(c.x - d.x, c.y - d.y)
    after  = math.hypot(a.x - c.x, a.y - c.y) + math.hypot(b.x - d.x, b.y - d.y)
    return after - before


def mutate(c: Chromosome, game_state: Any, progress: float = 0.5, diversity_score: float = 1.0):
    station_map = {s.id: s for s in game_state.stations}

    # Adaptive scaling: cool mutation rates as search matures; boost when diversity is low
    cooling = 1.0 - 0.5 * progress
    diversity_boost = 1.5 if diversity_score < 0.3 else 1.0
    scale = cooling * diversity_boost

    p_insert    = 0.15 * scale
    p_remove    = 0.10 * scale
    p_transfer  = 0.12 * scale
    p_invert    = 0.08 * scale
    p_direction = 0.10 * scale   # Loop direction flip
    p_resource  = 0.20 * cooling  # Resource mutation doesn't need diversity boost

    # Weight for type-alternation benefit in insertion scoring (in pixel-equivalent units)
    w_alt = 50.0

    for i in range(len(c.lines)):
        if not c.lines[i]: continue

        # 1. Insertion: best position by total line length + alternation benefit
        if random.random() < p_insert:
            available = [s.id for s in game_state.stations if s.id not in c.lines[i]]
            if available:
                s_new = random.choice(available)
                best_score = float('inf')
                best_idx = 0
                for idx in range(len(c.lines[i]) + 1):
                    temp_line = c.lines[i][:idx] + [s_new] + c.lines[i][idx:]
                    dist_cost = sum(
                        math.hypot(
                            station_map[temp_line[j]].x - station_map[temp_line[j + 1]].x,
                            station_map[temp_line[j]].y - station_map[temp_line[j + 1]].y
                        )
                        for j in range(len(temp_line) - 1)
                    )
                    alt_delta = _alternation_delta(c.lines[i], idx, s_new, station_map)
                    score = dist_cost + w_alt * alt_delta
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                c.lines[i].insert(best_idx, s_new)

        # 2. Removal
        if random.random() < p_remove and len(c.lines[i]) > 2:
            idx = random.randint(0, len(c.lines[i]) - 1)
            c.lines[i].pop(idx)

        # 3. Transfer
        if random.random() < p_transfer and len(c.lines[i]) > 1:
            idx = random.randint(0, len(c.lines[i]) - 1)
            s_transfer = c.lines[i].pop(idx)
            other_lines = [j for j in range(len(c.lines)) if j != i]
            if other_lines:
                target_line = random.choice(other_lines)
                if s_transfer not in c.lines[target_line]:
                    c.lines[target_line].append(s_transfer)

        # 4. Guided 2-opt inversion: sample candidate reversals, apply best improving move
        if random.random() < p_invert and len(c.lines[i]) > 2:
            best_delta = 0.0
            best_pair = None
            n = len(c.lines[i])
            for _ in range(5):  # Sample 5 candidates — O(1) delta computation each
                idx1 = random.randint(1, n - 2)
                idx2 = random.randint(idx1 + 1, n - 1)
                delta = _line_length_delta(c.lines[i], idx1, idx2, station_map)
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (idx1, idx2)
            if best_pair:
                idx1, idx2 = best_pair
                c.lines[i][idx1:idx2] = reversed(c.lines[i][idx1:idx2])
            else:
                # No improving move found; apply random inversion for exploration
                idx1 = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                c.lines[i][idx1:idx2] = reversed(c.lines[i][idx1:idx2])

        # 5. Loop direction mutation
        if random.random() < p_direction and c.is_loop[i]:
            c.loop_direction[i] = random.choice(_LOOP_DIRECTIONS)

    # Resource mutation: transfer train between lines
    if random.random() < p_resource:
        lines_with_trains = [i for i, t in enumerate(c.trains_per_line) if t > 0]
        if lines_with_trains:
            donor = random.choice(lines_with_trains)
            receptor = random.choice(range(len(c.trains_per_line)))
            c.trains_per_line[donor] -= 1
            c.trains_per_line[receptor] += 1

    c.repair()
    return c
