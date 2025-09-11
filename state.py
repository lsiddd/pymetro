# ./state.py

from typing import List, Any
import time
from config import CONFIG

class GameState:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self) -> None:
        """Reset game state to initial values"""
        self.paused: bool = False
        self.speed: int = 1
        self.score: int = 0
        self.week: int = 1
        self.day: int = 0
        self.stations: List[Any] = []
        self.lines: List[Any] = []
        self.trains: List[Any] = []
        self.passengers: List[Any] = []
        self.selected_line: int = 0
        self.available_lines: int = 3
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
        
        # --- CHANGE START ---
        self.difficulty_stage: int = 0 # For curriculum learning
        # --- CHANGE END ---

        # Apply city configuration
        city_config = CONFIG.CITIES[self.selected_city]
        self.bridges = city_config['bridges']
        self.max_lines = city_config['maxLines']
        
        # Mark pathfinding graph as dirty after reset
        try:
            from systems.pathfinding import mark_graph_dirty
            mark_graph_dirty()
        except ImportError:
            pass  # Pathfinding module might not be loaded yet
    
    def set_city(self, city: str) -> None:
        """Set selected city and update configuration"""
        self.selected_city = city
        city_config = CONFIG.CITIES[city]
        self.bridges = city_config['bridges']
        self.max_lines = city_config['maxLines']

# Global game state instance
game_state: GameState = GameState()