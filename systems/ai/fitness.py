# ./systems/ai/fitness.py
import math
import random
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional

class LiteStation:
    def __init__(self, id: int, type_: str, x: float, y: float):
        self.id = id
        self.type = type_
        self.x = x
        self.y = y
        self.passengers: List[str] = []  # Just store destination types
        self.capacity = 6  # Approximation

class LiteTrain:
    def __init__(self, line_id: int, start_station_idx: int, capacity: int, direction: int = 1):
        self.line_id = line_id
        self.passengers: List[str] = []
        self.capacity = capacity
        self.current_station_idx = start_station_idx
        self.next_station_idx = start_station_idx + 1
        self.direction = direction
        self.progress = 0.0
        self.state = 'WAITING'
        self.wait_timer = 0.0
        self.speed = 150.0  # Approx speed logic abstraction

class LiteSimulation:
    def __init__(self, stations_data, chromosome, seed: int):
        self.rand = random.Random(seed)
        self.is_loop = chromosome.is_loop[:]
        self.loop_direction = getattr(chromosome, 'loop_direction', ['both'] * len(chromosome.lines))

        # 1. Copy stations
        self.stations: Dict[int, LiteStation] = {}
        for s_data in stations_data:
            self.stations[s_data['id']] = LiteStation(s_data['id'], s_data['type'], s_data['x'], s_data['y'])
            # Inherit current passengers to simulate immediate queue impact
            for p_dest in s_data['passengers']:
                self.stations[s_data['id']].passengers.append(p_dest)

        # 2. Extract lines from chromosome
        self.lines: Dict[int, List[int]] = {}
        for line_id, line_stations in enumerate(chromosome.lines):
            self.lines[line_id] = line_stations[:]

        # 3. Build reachability index: for each (line, station), which destination types are reachable?
        self._reachable_types: Dict[int, Dict[int, set]] = {}
        for line_id, path in self.lines.items():
            reachable_from: Dict[int, set] = {}
            for i, sid in enumerate(path):
                if self.is_loop[line_id]:
                    other_ids = [path[j] for j in range(len(path)) if j != i]
                else:
                    other_ids = path[:i] + path[i+1:]
                reachable_from[sid] = {
                    self.stations[s].type for s in other_ids if s in self.stations
                }
            self._reachable_types[line_id] = reachable_from

        # 4. Build hop-distance index: hop_distance[station_id][dest_type] = min hops on any line.
        # Used during boarding so passengers prefer lines that bring them closer to their destination.
        self._hop_distance: Dict[int, Dict[str, int]] = {
            sid: defaultdict(lambda: 999) for sid in self.stations
        }
        self._build_hop_index()

        # 5. Create trains based on resource allocation and loop direction
        self.trains: List[LiteTrain] = []
        for line_id in range(len(chromosome.lines)):
            if not self.lines[line_id]: continue
            num_trains = chromosome.trains_per_line[line_id]
            carriages = chromosome.carriages_per_line[line_id]
            capacity = 6 * (1 + (carriages // max(1, num_trains))) if num_trains > 0 else 6

            direction_mode = self.loop_direction[line_id] if self.is_loop[line_id] else 'forward'

            for t_idx in range(num_trains):
                start_idx = self.rand.randint(0, max(0, len(self.lines[line_id]) - 1))

                if direction_mode == 'both':
                    direction = 1 if t_idx % 2 == 0 else -1
                elif direction_mode == 'backward':
                    direction = -1
                else:
                    direction = 1

                self.trains.append(LiteTrain(line_id, start_idx, capacity, direction))

        # Output metrics
        self.ticks_no_overcrowd = 0
        self.queue_sum = 0
        self.fragile_penalty = 0
        self.game_over = False
        self.game_over_tick = None          # which step triggered game over
        # Per-station overcrowding streak (in simulation steps)
        self._overcrowd_streak: Dict[int, int] = {sid: 0 for sid in self.stations}

        # Game over threshold: ~20 seconds of simulated time at 50 ms/step = 400 steps.
        # The real game uses roughly 20–30 s; 400 steps is a conservative threshold.
        self.GAME_OVER_THRESHOLD = 400

    def _build_hop_index(self):
        """
        BFS on the line graph to compute min hops from each station to each destination type.
        Transfers are modeled as one extra hop, encouraging direct routes.
        """
        # Build adjacency: station -> [(neighbor_station_id, line_id)]
        adjacency: Dict[int, List[tuple]] = defaultdict(list)
        for line_id, path in self.lines.items():
            n = len(path)
            if n < 2:
                continue
            for i in range(n - 1):
                adjacency[path[i]].append((path[i + 1], line_id))
                adjacency[path[i + 1]].append((path[i], line_id))
            if self.is_loop[line_id]:
                adjacency[path[-1]].append((path[0], line_id))
                adjacency[path[0]].append((path[-1], line_id))

        # For each station run BFS to find min hops to every other station
        for start_sid in self.stations:
            dist: Dict[int, int] = {start_sid: 0}
            q = deque([start_sid])
            while q:
                cur = q.popleft()
                for neighbor, _ in adjacency[cur]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[cur] + 1
                        q.append(neighbor)

            # Record min hops to each destination type
            for sid, hops in dist.items():
                if sid == start_sid:
                    continue
                dest_type = self.stations[sid].type
                if hops < self._hop_distance[start_sid][dest_type]:
                    self._hop_distance[start_sid][dest_type] = hops

    def tick(self, delta_time: float, H: int):
        # Probability pools
        types_available = list({s.type for s in self.stations.values()})

        # Simulation Loop
        for step in range(H):
            # Spawn logic (simplified)
            if step % 20 == 0 and len(types_available) > 1:
                spawn_station = self.rand.choice(list(self.stations.values()))
                dest = self.rand.choice(types_available)
                while dest == spawn_station.type:
                    dest = self.rand.choice(types_available)
                spawn_station.passengers.append(dest)

            # Train logic
            for t in self.trains:
                line_path = self.lines[t.line_id]
                if len(line_path) < 2: continue

                if t.state == 'WAITING':
                    t.wait_timer -= delta_time
                    if t.wait_timer <= 0:
                        t.state = 'MOVING'

                        # Next station logic
                        if self.is_loop[t.line_id]:
                            t.next_station_idx = (t.current_station_idx + t.direction) % len(line_path)
                        else:
                            if t.current_station_idx >= len(line_path) - 1:
                                t.direction = -1
                            elif t.current_station_idx <= 0:
                                t.direction = 1
                            t.next_station_idx = t.current_station_idx + t.direction
                        t.progress = 0.0
                elif t.state == 'MOVING':
                    s_current = self.stations[line_path[t.current_station_idx]]
                    s_next = self.stations[line_path[t.next_station_idx]]
                    dist = math.hypot(s_next.x - s_current.x, s_next.y - s_current.y) + 0.1

                    t.progress += t.speed * (delta_time / 1000.0)
                    if t.progress >= dist:
                        # Arrived
                        t.current_station_idx = t.next_station_idx
                        t.state = 'WAITING'
                        t.wait_timer = 300.0  # ms wait time

                        current_sid = line_path[t.current_station_idx]
                        s_arrived = self.stations[current_sid]

                        # Alight passengers whose destination matches this station type
                        remaining = [p for p in t.passengers if p != s_arrived.type]
                        t.passengers = remaining

                        # Board passengers whose destination is reachable on this line
                        # and whose hop-distance improves by boarding (vs. waiting for another line)
                        space = t.capacity - len(t.passengers)
                        if space > 0:
                            reachable = self._reachable_types.get(t.line_id, {}).get(current_sid, set())
                            s_queue = s_arrived.passengers
                            boardable = []
                            stranded = []
                            for p_dest in s_queue:
                                if p_dest not in reachable:
                                    stranded.append(p_dest)
                                    continue
                                # Check if this line reduces hop count to destination
                                # Find the closest station on this line that matches dest type
                                best_hops_on_line = min(
                                    (idx for idx, sid in enumerate(line_path)
                                     if self.stations[sid].type == p_dest),
                                    default=None
                                )
                                if best_hops_on_line is None:
                                    stranded.append(p_dest)
                                    continue
                                # Hops from current position on line to destination station
                                cur_idx = t.current_station_idx
                                n = len(line_path)
                                if self.is_loop[t.line_id]:
                                    fwd = (best_hops_on_line - cur_idx) % n
                                    bwd = (cur_idx - best_hops_on_line) % n
                                    line_hops = min(fwd, bwd) if self._loop_is_bidirectional(t.line_id) else fwd
                                else:
                                    line_hops = abs(best_hops_on_line - cur_idx)

                                # Compare with global min hops from this station to dest
                                global_min = self._hop_distance[current_sid].get(p_dest, 999)
                                if line_hops <= global_min + 1:  # allow 1 hop slack for transfers
                                    boardable.append(p_dest)
                                else:
                                    stranded.append(p_dest)

                            boarding = boardable[:space]
                            t.passengers.extend(boarding)
                            s_arrived.passengers = stranded + boardable[space:]

                        # Stranded passengers with no reachable destination on any line
                        for p_dest in s_arrived.passengers:
                            on_any_line = any(
                                p_dest in self._reachable_types.get(lid, {}).get(current_sid, set())
                                for lid in self._reachable_types
                            )
                            if not on_any_line:
                                self.fragile_penalty += 1

            # Metrics gathering
            any_overcrowded = False
            for s in self.stations.values():
                queue_len = len(s.passengers)
                self.queue_sum += queue_len

                if queue_len > s.capacity:
                    any_overcrowded = True
                    self._overcrowd_streak[s.id] += 1
                    if self._overcrowd_streak[s.id] >= self.GAME_OVER_THRESHOLD:
                        self.game_over = True
                        self.game_over_tick = step
                else:
                    self._overcrowd_streak[s.id] = 0

                # Fragile Penalty: above 70% capacity
                if queue_len > s.capacity * 0.7:
                    self.fragile_penalty += 1

            if not any_overcrowded:
                self.ticks_no_overcrowd += 1

            if self.game_over:
                break

    def _loop_is_bidirectional(self, line_id: int) -> bool:
        return self.loop_direction[line_id] == 'both'


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


def calculate_fitness(chromosome: Any, stations_data: List[Dict], seed: int, H: int) -> float:
    """
    Evaluates the chromosome via headless simulation.
    F = w1*ticks - w2*queue_sum - w3*fragility - w4*cost + strategic_bonuses
    """
    sim = LiteSimulation(stations_data, chromosome, seed)
    sim.tick(delta_time=50.0, H=H)

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

    # Stations that ARE of the hottest type (passengers want to go there)
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
        # Collect all station IDs co-located with this special on any line
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

    # Game-over penalty: a solution that causes any station to overflow for
    # GAME_OVER_THRESHOLD consecutive steps is fundamentally broken.
    # Apply a steep penalty proportional to how early the collapse happened:
    # earlier failure = worse score. Solutions that survive the full horizon
    # are always preferred over solutions that don't.
    if sim.game_over:
        survival_ratio = (sim.game_over_tick or 0) / H  # 0.0 = immediate failure
        game_over_penalty = 10_000.0 * (1.0 - survival_ratio)
        total -= game_over_penalty

    return total
