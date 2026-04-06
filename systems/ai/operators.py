# ./systems/ai/operators.py
import random
import math
from typing import List, Any
from config import STATION_TYPES
from .chromosome import Chromosome

def generate_initial_population(N: int, game_state: Any) -> List[Chromosome]:
    population = []
    
    # Weights for station types
    special_types = STATION_TYPES.special_types()
    weights = {}
    for station in game_state.stations:
        weight = 10 if station.type in special_types else 1
        weight += len(station.passengers) * 2  # high queue -> high prior
        weights[station] = weight

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
            
            # Neartest neighbor ordering
            if selected:
                ordered = [selected.pop(0)]
                while selected:
                    last = ordered[-1]
                    next_s = min(selected, key=lambda s: math.hypot(s.x - last.x, s.y - last.y))
                    selected.remove(next_s)
                    ordered.append(next_s)
                c.lines[i] = ordered
            
            if random.random() < 0.4 and len(c.lines[i]) > 2:
                c.is_loop[i] = True

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
        
    child = Chromosome(len(parent_a.lines))
    child.trains_per_line = parent_a.trains_per_line[:]
    child.carriages_per_line = parent_a.carriages_per_line[:]
    child.is_loop = parent_a.is_loop[:]
    
    # EAX adapted logic
    for i in range(len(parent_a.lines)):
        line_a = parent_a.lines[i]
        line_b = parent_b.lines[i]
        
        if not line_a or not line_b:
            child.lines[i] = line_a[:] if line_a else line_b[:]
            continue
            
        # Simplified EAX: 2-point crossover on station lists to avoid complex graph disjoint loops
        if len(line_a) > 2 and len(line_b) > 2:
            pt1 = random.randint(0, len(line_a) - 1)
            pt2 = random.randint(pt1, len(line_a))
            segment = line_a[pt1:pt2]
            
            new_line = segment[:]
            for s in line_b:
                if s not in new_line:
                    new_line.append(s)
            child.lines[i] = new_line
        else:
            child.lines[i] = line_a[:] if random.random() < 0.5 else line_b[:]
    
    child.repair()
    return child

def mutate(c: Chromosome, game_state: Any):
    for i in range(len(c.lines)):
        if not c.lines[i]: continue
        
        # 1. Insertion
        if random.random() < 0.15:
            available = [s for s in game_state.stations if s not in c.lines[i]]
            if available:
                s_new = random.choice(available)
                best_cost = float('inf')
                best_idx = 0
                for idx in range(len(c.lines[i]) + 1):
                    temp_line = c.lines[i][:idx] + [s_new] + c.lines[i][idx:]
                    cost = 0
                    for j in range(len(temp_line) - 1):
                        cost += math.hypot(temp_line[j].x - temp_line[j+1].x, temp_line[j].y - temp_line[j+1].y)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                c.lines[i].insert(best_idx, s_new)
                
        # 2. Removal
        if random.random() < 0.10 and len(c.lines[i]) > 2:
            idx = random.randint(0, len(c.lines[i]) - 1)
            c.lines[i].pop(idx)
            
        # 3. Transfer
        if random.random() < 0.12 and len(c.lines[i]) > 1:
            idx = random.randint(0, len(c.lines[i]) - 1)
            s_transfer = c.lines[i].pop(idx)
            other_lines = [j for j in range(len(c.lines)) if j != i]
            if other_lines:
                target_line = random.choice(other_lines)
                if s_transfer not in c.lines[target_line]:
                    c.lines[target_line].append(s_transfer)
                    
        # 4. Inversion
        if random.random() < 0.08 and len(c.lines[i]) > 2:
            idx1 = random.randint(0, len(c.lines[i]) - 1)
            idx2 = random.randint(0, len(c.lines[i]) - 1)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            c.lines[i][idx1:idx2] = reversed(c.lines[i][idx1:idx2])
            
    # Resource Mut
    if random.random() < 0.20:
        lines_with_trains = [i for i, t in enumerate(c.trains_per_line) if t > 0]
        if lines_with_trains:
            donor = random.choice(lines_with_trains)
            receptor = random.choice(range(len(c.trains_per_line)))
            c.trains_per_line[donor] -= 1
            c.trains_per_line[receptor] += 1
            
    c.repair()
    return c
