# ./systems/ai/ga.py
import random
import time
import threading
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from .operators import generate_initial_population, generate_heuristic_seeds, tournament_selection, edge_assembly_crossover, mutate
from .fitness import calculate_fitness
from .ga_cache import make_signature, load_cache, save_cache

def _eval_wrapper(args):
    chromosome, stations_data, seed, H, snapshot_week = args
    return calculate_fitness(chromosome, stations_data, seed, H, snapshot_week)

def _compute_diversity(population):
    unique = len({tuple(tuple(l) for l in c.lines) for c in population})
    return unique / len(population)

class GeneticAlgorithmTask(threading.Thread):
    def __init__(self, game_state: Any, on_complete=None, pop_size=200, gen_max=50, sim_horizon=3000):
        super().__init__(daemon=True)
        self.game_state = game_state
        self.on_complete = on_complete
        self.pop_size = pop_size
        self.gen_max = gen_max
        self.sim_horizon = sim_horizon

        self.best_chromosome = None
        self.progress = 0.0
        self.generation = 0
        self.is_running = True

    def run(self):
        print("--- Rodando Algoritmo Genético ---")
        G_max = self.gen_max
        N = self.pop_size
        H = self.sim_horizon
        budget_sec = 300

        start_time = time.time()

        seed_atual = 42

        # Snapshot stations for multiprocessing workers
        lite_stations_data = [
            {
                'id': s.id,
                'type': s.type,
                'x': s.x,
                'y': s.y,
                'passengers': [p.destination for p in s.passengers],
                'is_interchange': s.is_interchange,
            }
            for s in self.game_state.stations
        ]
        snapshot_week = self.game_state.week

        # Warm-start: inject cached chromosomes if map hasn't changed
        sig = make_signature(self.game_state.stations)
        cached = load_cache(sig)
        population = generate_initial_population(N, self.game_state)

        # Inject heuristic seeds (known-good MiniMetro strategies)
        seeds = generate_heuristic_seeds(self.game_state, n=20)
        n_seeds = min(len(seeds), N // 10)
        print(f"[AG] Injetando {n_seeds} cromossomos heurísticos.")
        for i in range(n_seeds):
            population[i] = seeds[i]

        if cached:
            n_inject = min(len(cached), N // 4)
            print(f"[AG] Carregando {n_inject} cromossomos do cache.")
            offset = n_seeds
            for i in range(n_inject):
                if offset + i < N:
                    population[offset + i] = cached[i].copy()

        best_global_fitness = float('-inf')
        self.best_chromosome = None
        stagnation_counter = 0

        # Keep executor alive across all generations to avoid repeated spawn/teardown overhead
        with ProcessPoolExecutor() as executor:
            for g in range(1, G_max + 1):
                if not self.is_running:
                    break

                self.generation = g
                self.progress = g / G_max

                # Short horizon early (cheap exploration), long horizon late (precise evaluation)
                if g <= 5:
                    H_gen = 1000
                elif g <= G_max - 10:
                    H_gen = H
                else:
                    H_gen = 5000

                args_list = [(p, lite_stations_data, seed_atual, H_gen, snapshot_week) for p in population]
                fitnesses = list(executor.map(_eval_wrapper, args_list))

                gen_best_idx = fitnesses.index(max(fitnesses))
                gen_best_fit = fitnesses[gen_best_idx]

                print(f"[AG] Geração {g}/{G_max} - Melhor deste ciclo: {gen_best_fit:.2f} (H={H_gen})")

                if gen_best_fit > best_global_fitness:
                    best_global_fitness = gen_best_fit
                    self.best_chromosome = population[gen_best_idx].copy()
                    stagnation_counter = 0
                    print(f"      -> Novo Melhor Global Encontrado! {best_global_fitness:.2f}")
                else:
                    stagnation_counter += 1

                # Check Time Budget
                if (time.time() - start_time) > budget_sec:
                    print(f"[AG] Budget de tempo esgotado ({budget_sec}s). Parando busca.")
                    break

                # Next generation
                if g == G_max: break

                # Elitismo: Os 2 melhores avançam diretamente
                elite_count = 2
                sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
                P_next = [sorted_pop[i][0].copy() for i in range(elite_count)]

                # Diversity injection on stagnation: replace worst with random + perturbations of best
                if stagnation_counter >= 5:
                    print(f"[AG] Estagnação detectada ({stagnation_counter} gens). Injetando diversidade.")
                    num_replace = int(0.3 * N)
                    new_random = generate_initial_population(num_replace // 2, self.game_state)
                    new_perturb = []
                    for _ in range(num_replace - num_replace // 2):
                        candidate = self.best_chromosome.copy()
                        for _ in range(random.randint(3, 5)):
                            candidate = mutate(candidate, self.game_state)
                        new_perturb.append(candidate)
                    injected = new_random + new_perturb
                    # Replace worst slots (beyond elites) in sorted_pop
                    sorted_list = list(sorted_pop)
                    worst_start = len(sorted_list) - num_replace
                    for i, candidate in enumerate(injected):
                        sorted_list[worst_start + i] = (candidate, float('-inf'))
                    sorted_pop = sorted_list
                    stagnation_counter = 0

                # Compute diversity for adaptive mutation
                diversity_score = _compute_diversity(population)
                gen_progress = g / G_max

                while len(P_next) < N:
                    if not self.is_running: break
                    pai_A = tournament_selection(population, fitnesses)
                    pai_B = tournament_selection(population, fitnesses)
                    filho = edge_assembly_crossover(pai_A, pai_B)
                    filho = mutate(filho, self.game_state, progress=gen_progress, diversity_score=diversity_score)
                    P_next.append(filho)

                population = P_next
                seed_atual = hash(str(seed_atual)) % (2**32)

        # Persist top chromosomes for warm-start on next invocation
        if self.best_chromosome:
            top = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
            save_cache([c.copy() for c, _ in top[:10]], sig)

        print("--- Algoritmo Genético Finalizado ---")
        if self.on_complete and self.best_chromosome:
            self.on_complete(self.best_chromosome)
