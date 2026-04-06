# ./state.py

from typing import List, Any, Optional
import time
from config import CONFIG

class GameState:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset game state to initial values"""
        self.paused: bool = False
        self.spawn_stations_enabled: bool = True
        self.speed: float = 1.0
        self.score: int = 0
        self.week: int = 1
        self.day: int = 0
        self.stations: List[Any] = []
        self.lines: List[Any] = []
        self.trains: List[Any] = []
        self.passengers: List[Any] = []
        self.selected_line: int = 0
        self.max_lines: int = 5
        self.bridges: int = 2
        self.carriages: int = 0
        self.interchanges: int = 0
        self.available_trains: int = 3
        self.game_over: bool = False
        self.selected_city: str = 'london'
        self.rivers: List[Any] = []
        self.last_spawn_time: float = time.time() * 1000
        self.last_station_spawn_time: float = time.time() * 1000
        self.week_start_time: float = time.time() * 1000
        self.station_id_counter: int = 0
        self.train_id_counter: int = 0
        self.passengers_delivered: int = 0
        self.game_start_time: float = time.time() * 1000

        self.camera_zoom: float = 1.0   # Decreases over weeks → visual zoom-out

        # Headless simulation clock (used by GA fitness evaluation)
        self.headless: bool = False
        self.sim_time_ms: float = 0.0

        # Apply city configuration
        city_config = CONFIG.CITIES[self.selected_city]
        self.bridges = city_config['bridges']
        self.max_lines = city_config['maxLines']
        self.available_lines: int = 3  # Player starts with 3 lines; more unlock via weekly upgrades
        
    
    def set_city(self, city: str) -> None:
        """Set selected city and update configuration"""
        self.selected_city = city
        city_config = CONFIG.CITIES[city]
        self.bridges = city_config['bridges']
        self.max_lines = city_config['maxLines']

# Global game state instance
game_state: GameState = GameState()


def now_ms() -> float:
    """Current time in milliseconds.

    In normal mode returns wall-clock time. In headless mode (GA fitness
    evaluation) returns the virtual simulation clock so that all timing logic
    in the engine — spawn rates, overcrowd timers, passenger patience — runs
    on controllable, deterministic virtual time instead of real wall time.
    """
    return game_state.sim_time_ms if game_state.headless else time.time() * 1000