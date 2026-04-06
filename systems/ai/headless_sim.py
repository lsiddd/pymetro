# ./systems/ai/headless_sim.py
"""
Runs the real game engine in headless mode for GA fitness evaluation.

Each HeadlessSimulation resets the global game_state, reconstructs stations,
lines, and trains from the snapshot + chromosome, then drives Game.update()
with a virtual clock. All timing in the engine (spawn rates, overcrowd timers,
passenger patience) operates on sim_time_ms instead of wall time.

Safe for use in ProcessPoolExecutor workers: each worker process has its own
copy of game_state (fork isolation), and evaluations within a worker are
sequential, so resetting the global is race-free.
"""

import random
from typing import List, Dict, Any

from config import CONFIG
from state import game_state, now_ms

# Dimensions used for station placement boundary checks inside Game.update.
# Must be large enough that no station is considered "off-screen".
_SCREEN_W = 1280
_SCREEN_H = 800


class HeadlessSimulation:
    def __init__(
        self,
        stations_data: List[Dict],
        chromosome: Any,
        seed: int,
        snapshot_week: int = 1,
    ) -> None:
        # Late imports keep the dependency on pygame/game classes inside worker
        # processes; the env-var trick for SDL_VIDEODRIVER is not needed because
        # we never call any pygame drawing function from this path.
        from components.line import Line
        from components.passenger import Passenger
        from components.train import Train
        from systems.game import Game
        from systems.pathfinding import mark_graph_dirty

        rand = random.Random(seed)

        # ------------------------------------------------------------------ #
        # 1. Reset game state and switch to virtual clock
        # ------------------------------------------------------------------ #
        game_state.reset()
        game_state.headless = True
        game_state.spawn_stations_enabled = False
        game_state.week = snapshot_week

        # Zero the virtual clock and sync all timer fields to it so that
        # spawn-rate comparisons work correctly from the first update step.
        game_state.sim_time_ms = 0.0
        game_state.last_spawn_time = 0.0
        game_state.last_station_spawn_time = 0.0
        game_state.game_start_time = 0.0
        # Push week_start far ahead so week progression never fires during the
        # simulation (max H=5000 steps * 50 ms = 250 s < WEEK_DURATION * 10).
        game_state.week_start_time = CONFIG.WEEK_DURATION * 10.0

        # ------------------------------------------------------------------ #
        # 2. Create Line objects (mirrors Game.init_game)
        # ------------------------------------------------------------------ #
        game_state.lines = []
        for i in range(game_state.max_lines):
            game_state.lines.append(
                Line(CONFIG.LINE_COLORS[i % len(CONFIG.LINE_COLORS)], i)
            )

        # ------------------------------------------------------------------ #
        # 3. Restore Station objects with original IDs from snapshot
        #
        # Station.__new__ bypasses __init__ to avoid auto-incrementing
        # station_id_counter and triggering mark_graph_dirty before lines
        # are configured.
        # ------------------------------------------------------------------ #
        from components.station import Station

        station_map: Dict[int, Any] = {}
        for s_data in stations_data:
            st = object.__new__(Station)
            st.id = s_data["id"]
            st.x = s_data["x"]
            st.y = s_data["y"]
            st.type = s_data["type"]
            st.is_interchange = s_data.get("is_interchange", False)
            st.passengers = []
            st.overcrowd_start_time = None
            st.connection_animation = None
            st.delivery_animation = None
            st.animate_upgrade = None
            game_state.stations.append(st)
            station_map[st.id] = st

        # ------------------------------------------------------------------ #
        # 4. Apply chromosome: populate lines with stations
        # ------------------------------------------------------------------ #
        for i, station_ids in enumerate(chromosome.lines):
            if not station_ids:
                continue
            line = game_state.lines[i]
            for sid in station_ids:
                if sid in station_map:
                    line.add_station(station_map[sid])
            # Close loop by repeating the first station at the end, matching
            # the real game's loop representation (stations[0] == stations[-1]).
            if (
                chromosome.is_loop[i]
                and len(station_ids) >= 3
                and station_ids[0] in station_map
            ):
                line.add_station(station_map[station_ids[0]])

        # Rebuild pathfinding graph from the configured lines before creating
        # passengers (Passenger.__init__ calls recalculate_path immediately).
        mark_graph_dirty()

        # ------------------------------------------------------------------ #
        # 5. Restore passengers from snapshot
        # ------------------------------------------------------------------ #
        for s_data in stations_data:
            st = station_map[s_data["id"]]
            for dest in s_data["passengers"]:
                p = Passenger(st, dest)
                st.passengers.append(p)
                game_state.passengers.append(p)
            # Trigger overcrowd tracking for stations that start over capacity.
            st.check_overcrowd()

        # ------------------------------------------------------------------ #
        # 6. Create trains from chromosome
        #
        # Train.__init__ calls process_passengers(line.stations[0]) so trains
        # must be created after passengers are on the stations.
        # ------------------------------------------------------------------ #
        for i in range(len(chromosome.lines)):
            line = game_state.lines[i]
            if not line.active:
                continue
            n_trains = chromosome.trains_per_line[i]
            n_carriages = chromosome.carriages_per_line[i]
            carriages_per_train = n_carriages // max(1, n_trains)
            for _ in range(n_trains):
                t = Train(line)
                t.carriage_count = carriages_per_train
                game_state.trains.append(t)
                line.trains.append(t)

        # ------------------------------------------------------------------ #
        # 7. Wire up Game instance without going through init_game
        # ------------------------------------------------------------------ #
        self._game = Game()
        self._game.initialized = True

        # Simulation metrics (same interface as LiteSimulation)
        self.ticks_no_overcrowd: int = 0
        self.queue_sum: int = 0
        self.fragile_penalty: int = 0
        self.game_over: bool = False
        self.game_over_tick: int | None = None

    # ---------------------------------------------------------------------- #

    def run(self, delta_time: float = 50.0, steps: int = 800) -> None:
        """Advance the simulation for *steps* ticks of *delta_time* ms each."""
        for step in range(steps):
            game_state.sim_time_ms += delta_time
            result = self._game.update(delta_time, _SCREEN_W, _SCREEN_H)

            # Collect per-step metrics before checking termination.
            any_overcrowded = False
            for s in game_state.stations:
                q = len(s.passengers)
                self.queue_sum += q
                if q > s.capacity:
                    any_overcrowded = True
                if q > s.capacity * 0.7:
                    self.fragile_penalty += 1

            if not any_overcrowded:
                self.ticks_no_overcrowd += 1

            # game_state.game_over is set inside Game.update → check_game_over.
            if game_state.game_over or result == "game_over":
                self.game_over = True
                self.game_over_tick = step
                break
