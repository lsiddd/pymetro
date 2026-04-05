# ./systems/ai/fitness.py
import math
import random
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
    def __init__(self, line_id: int, start_station_idx: int, capacity: int):
        self.line_id = line_id
        self.passengers: List[str] = []
        self.capacity = capacity
        self.current_station_idx = start_station_idx
        self.next_station_idx = start_station_idx + 1
        self.direction = 1
        self.progress = 0.0
        self.state = 'WAITING'
        self.wait_timer = 0.0
        self.speed = 150.0  # Approx speed logic abstraction

class LiteSimulation:
    def __init__(self, stations_data, chromosome, seed: int):
        self.rand = random.Random(seed)
        self.is_loop = chromosome.is_loop[:]
        
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

        # 3. Create trains based on resource allocation
        self.trains: List[LiteTrain] = []
        for line_id in range(len(chromosome.lines)):
            if not self.lines[line_id]: continue
            num_trains = chromosome.trains_per_line[line_id]
            carriages = chromosome.carriages_per_line[line_id]
            
            capacity = 6 * (1 + (carriages // max(1, num_trains))) if num_trains > 0 else 6
            
            for _ in range(num_trains):
                start_idx = self.rand.randint(0, max(0, len(self.lines[line_id]) - 1))
                self.trains.append(LiteTrain(line_id, start_idx, capacity))

        # Output metrics
        self.ticks_no_overcrowd = 0
        self.queue_sum = 0
        self.fragile_penalty = 0

    def tick(self, delta_time: float, H: int):
        # Probability pools
        types_available = list({s.type for s in self.stations.values()})
        
        # 4. Simulation Loop
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
                        t.wait_timer = 300.0  # ms Wait time

                        # Board / Alight passengers
                        # Alight if destination matches or transfer required (simple)
                        remaining = []
                        alighted = 0
                        for p in t.passengers:
                            if p == s_next.type:
                                alighted += 1
                            else:
                                remaining.append(p)
                        t.passengers = remaining

                        # Board
                        boarded = 0
                        space = t.capacity - len(t.passengers)
                        if space > 0:
                            s_queue = s_next.passengers
                            boarding = s_queue[:space]
                            t.passengers.extend(boarding)
                            s_next.passengers = s_queue[space:]

            # Metrics gathering
            any_overcrowded = False
            for s in self.stations.values():
                queue_len = len(s.passengers)
                self.queue_sum += queue_len
                
                if queue_len > s.capacity:
                    any_overcrowded = True
                
                # Fragile Penalty: above 70% capacity
                if queue_len > s.capacity * 0.7:
                    self.fragile_penalty += 1

            if not any_overcrowded:
                self.ticks_no_overcrowd += 1


def calculate_fitness(chromosome: Any, stations_data: List[Dict], seed: int, H: int) -> float:
    """
    Evaluates the chromosome via headless simulation.
    F = w1*ticks - w2*queue_sum - w3*fragility - w4*cost
    """
    sim = LiteSimulation(stations_data, chromosome, seed)
    
    # Run 50 ms ticks for H iterations
    sim.tick(delta_time=50.0, H=H)

    w1 = 10.0
    w2 = 1.0
    w3 = 5.0
    w4 = 50.0
    w5 = 500.0 # Penalty per active line without trains
    
    # Penalidades configuráveis para tamanho de linha
    w_length = 0.05          # Fator de penalidade por pixel (ex: 1000px = -50 pontos)
    w_excess_stations = 25.0 # Penalidade por cada estação além de 6 numa mesma linha
    
    cost = sum(chromosome.trains_per_line) + sum(chromosome.carriages_per_line)
    empty_lines = sum(1 for i in range(len(chromosome.lines)) if len(chromosome.lines[i]) >= 2 and chromosome.trains_per_line[i] == 0)
    
    total_physical_length = 0.0
    excess_stations_count = 0
    station_map = {s['id']: s for s in stations_data}
    
    for i in range(len(chromosome.lines)):
        line_path = chromosome.lines[i]
        n_stations = len(line_path)
        if n_stations >= 2:
            # Calcula distância física
            for j in range(n_stations - 1):
                s1, s2 = station_map[line_path[j]], station_map[line_path[j+1]]
                total_physical_length += math.hypot(s2['x'] - s1['x'], s2['y'] - s1['y'])
            
            if chromosome.is_loop[i]:
                s1, s2 = station_map[line_path[-1]], station_map[line_path[0]]
                total_physical_length += math.hypot(s2['x'] - s1['x'], s2['y'] - s1['y'])
                
            # Verifica limite de estações ideal
            if n_stations > 6:
                excess_stations_count += (n_stations - 6)
    
    fitness = (w1 * sim.ticks_no_overcrowd) - (w2 * sim.queue_sum) - (w3 * sim.fragile_penalty) - (w4 * cost) - (w5 * empty_lines) - (w_length * total_physical_length) - (w_excess_stations * excess_stations_count)
    return fitness
