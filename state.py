import time
from config import CONFIG

class GameState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset game state to initial values"""
        self.paused = False
        self.speed = 1
        self.score = 0
        self.week = 1
        self.day = 0
        self.stations = []
        self.lines = []
        self.trains = []
        self.passengers = []
        self.selected_line = 0
        self.available_lines = 3
        self.max_lines = 5
        self.bridges = 2
        self.carriages = 0
        self.interchanges = 0
        self.available_trains = 3
        self.game_over = False
        self.selected_city = 'london'
        self.rivers = []
        self.last_spawn_time = time.time() * 1000
        self.last_station_spawn_time = time.time() * 1000
        self.week_start_time = time.time() * 1000
        self.station_id_counter = 0
        self.train_id_counter = 0
        self.passengers_delivered = 0
        self.game_start_time = time.time() * 1000
        
        # Apply city configuration
        city_config = CONFIG.CITIES[self.selected_city]
        self.bridges = city_config['bridges']
        self.max_lines = city_config['maxLines']
    
    def set_city(self, city):
        """Set selected city and update configuration"""
        self.selected_city = city
        city_config = CONFIG.CITIES[city]
        self.bridges = city_config['bridges']
        self.max_lines = city_config['maxLines']

# Global game state instance
game_state = GameState()