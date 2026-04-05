# ./systems/ai/ga.py
import time
import threading
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from .operators import generate_initial_population, tournament_selection, edge_assembly_crossover, mutate
from .fitness import calculate_fitness

def _eval_wrapper(args):
    chromosome, stations_data, seed, H = args
    return calculate_fitness(chromosome, stations_data, seed, H)

class GeneticAlgorithmTask(threading.Thread):
    def __init__(self, game_state: Any, on_complete=None):
        super().__init__(daemon=True)
        self.game_state = game_state
        self.on_complete = on_complete
        
        self.best_chromosome = None
        self.progress = 0.0
        self.generation = 0
        self.is_running = True

    def run(self):
        print("--- Rodando Algoritmo Genético ---")
        G_max = 50 # Reduced to 30 to help match python runtime performance constraints
        N = 200    # Reduced pop to fit 60s
        H = 3000    # Reduced horizon ticks
        budget_sec = 120
        start_time = time.time()
        
        seed_atual = 42
        
        # Prepare lite stations data for multiprocessing
        lite_stations_data = [
            {
                'id': s.id,
                'type': s.type,
                'x': s.x,
                'y': s.y,
                'passengers': [p.destination for p in s.passengers]
            }
            for s in self.game_state.stations
        ]
        
        # Initial Pop
        population = generate_initial_population(N, self.game_state)
        
        best_global_fitness = float('-inf')
        self.best_chromosome = None
        stagnation_counter = 0

        for g in range(1, G_max + 1):
            if not self.is_running:
                break
                
            self.generation = g
            self.progress = g / G_max
            
            # Evaluate
            H_gen = 1000 if g <= G_max - 10 else H
            args_list = [(p, lite_stations_data, seed_atual, H_gen) for p in population]
            
            with ProcessPoolExecutor() as executor:
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
            
            # Elitismo: Os 2 melhores avançam diretamente sem sofrer mutações
            P_next = []
            elite_count = 2
            sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
            for i in range(elite_count):
                P_next.append(sorted_pop[i][0].copy())
                
            # Injeção de diversidade se houver estagnação
            if stagnation_counter >= 5:
                print(f"[AG] Estagnação detectada. Injetando diversidade.")
                num_replace = int(0.3 * N)
                new_individuals = generate_initial_population(num_replace, self.game_state)
                P_next.extend(new_individuals)
                stagnation_counter = 0
                
            while len(P_next) < N:
                if not self.is_running: break
                pai_A = tournament_selection(population, fitnesses)
                pai_B = tournament_selection(population, fitnesses)
                filho = edge_assembly_crossover(pai_A, pai_B)
                filho = mutate(filho, self.game_state)
                P_next.append(filho)
                
            population = P_next
            seed_atual = hash(str(seed_atual)) % (2**32)
            
        print("--- Algoritmo Genético Finalizado ---")
        if self.on_complete and self.best_chromosome:
            # Emite callback quando termina
            self.on_complete(self.best_chromosome)
