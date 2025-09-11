# ./systems/game.py

import pygame
import math
import time
import random
from config import CONFIG, STATION_TYPES
from components.station import Station
from components.passenger import Passenger
from components.line import Line
from components.train import Train
from components.river import River

class Game:
    def __init__(self):
        self.initialized = False
    
    # --- CHANGE START ---
    def init_game(self, screen_width, screen_height, difficulty_stage=0):
    # --- CHANGE END ---
        """Initialize new game"""
        from state import game_state
        
        game_state.reset()
        
        # --- CHANGE START ---
        # Set difficulty for this episode
        game_state.difficulty_stage = difficulty_stage
        # --- CHANGE END ---

        # Create lines
        game_state.lines = []
        for i in range(game_state.max_lines):
            color_hex = CONFIG.LINE_COLORS[i % len(CONFIG.LINE_COLORS)]
            game_state.lines.append(Line(color_hex, i))
        
        # Create rivers based on selected city
        self.create_rivers(screen_width, screen_height)
        
        # Create initial stations
        self.create_initial_stations(screen_width, screen_height)
        
        self.initialized = True
    
    def create_rivers(self, width, height):
        """Create rivers for the selected city"""
        from state import game_state
        
        game_state.rivers = []
        
        if game_state.selected_city == 'london':
            # Thames-like river
            points = [
                {'x': 0, 'y': height * 0.65},
                {'x': width * 0.3, 'y': height * 0.55},
                {'x': width * 0.7, 'y': height * 0.6},
                {'x': width, 'y': height * 0.5},
                {'x': width, 'y': height * 0.6},
                {'x': width * 0.7, 'y': height * 0.7},
                {'x': width * 0.3, 'y': height * 0.65},
                {'x': 0, 'y': height * 0.75}
            ]
            game_state.rivers.append(River(points))
    
    def create_initial_stations(self, width, height):
        """Create initial stations"""
        from state import game_state
        
        types = STATION_TYPES.basic_types()
        center_x = width // 2
        center_y = height // 2
        spread = min(width, height) * 0.2
        min_distance = 120
        
        for station_type in types:
            attempts = 0
            max_attempts = 200
            
            while attempts < max_attempts:
                angle = random.random() * math.pi * 2
                radius = spread * (0.5 + random.random() * 0.5)
                x = center_x + math.cos(angle) * radius
                y = center_y + math.sin(angle) * radius
                
                # Check if position is valid
                too_close = any(math.hypot(s.x - x, s.y - y) < min_distance for s in game_state.stations)
                in_river = any(river.contains(x, y) for river in game_state.rivers)
                
                if not too_close and not in_river:
                    game_state.stations.append(Station(x, y, station_type))
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback: place near center
                offset_x = (random.random() - 0.5) * 50
                offset_y = (random.random() - 0.5) * 50
                game_state.stations.append(Station(center_x + offset_x, center_y + offset_y, station_type))
    
    def update(self, delta_time, screen_width, screen_height):
        """Update game logic"""
        from state import game_state
        
        
        if not self.initialized or game_state.game_over or game_state.paused:
            return

        current_time = time.time() * 1000
        
        # Update week/day progression
        elapsed = current_time - game_state.week_start_time
        game_state.day = int(elapsed / (CONFIG.WEEK_DURATION / 7)) % 7
        
        if elapsed > CONFIG.WEEK_DURATION:
            game_state.week += 1
            game_state.week_start_time = current_time
            return 'show_upgrades'
        
        # --- CHANGE START ---
        # Progressive difficulty scaling using curriculum learning stage
        difficulty_settings = CONFIG.DIFFICULTY_LEVELS[game_state.difficulty_stage]
        difficulty_multiplier = CONFIG.DIFFICULTY_SCALE_FACTOR ** (game_state.week - 1)
        
        passenger_multiplier = difficulty_settings['passenger_spawn_multiplier']
        station_multiplier = difficulty_settings['station_spawn_multiplier']

        current_spawn_rate = CONFIG.BASE_SPAWN_RATE * difficulty_multiplier * passenger_multiplier
        current_station_spawn_rate = CONFIG.BASE_STATION_SPAWN_RATE * difficulty_multiplier * station_multiplier
        # --- CHANGE END ---
        
        # Spawn passengers
        if current_time - game_state.last_spawn_time > current_spawn_rate / game_state.speed:
            self.spawn_passenger()
            game_state.last_spawn_time = current_time
        
        # Spawn stations
        if current_time - game_state.last_station_spawn_time > current_station_spawn_rate / game_state.speed:
            self.spawn_station(screen_width, screen_height)
            game_state.last_station_spawn_time = current_time
        
        # Update trains
        for train in game_state.trains[:]: # Use a copy in case a train removes itself
            train.update(delta_time)

        # --- CHANGE START ---
        # Finalize deletion of empty lines that were marked
        for line in game_state.lines:
            if line.marked_for_deletion and not line.trains:
                line.clear_line()
        # --- CHANGE END ---

        # Check game over condition
        if self.check_game_over():
            game_state.game_over = True
            game_state.paused = True
            return 'game_over'
        
        return None
    
    def spawn_passenger(self):
        """Spawn a new passenger at a random station"""
        from state import game_state
        
        if len(game_state.stations) < 2:
            return
        
        station = random.choice(game_state.stations)
        
        # Find available destination types (not the same as current station)
        available_types = [s.type for s in game_state.stations if s.type != station.type]
        available_types = list(set(available_types))  # Remove duplicates
        
        if available_types:
            destination = random.choice(available_types)
            passenger = Passenger(station, destination)
            station.add_passenger(passenger)
            game_state.passengers.append(passenger)
    
    def spawn_station(self, screen_width, screen_height):
        """Spawn a new station"""
        from state import game_state
        
        margin = 80
        min_distance = 120
        max_attempts = 500
        
        for attempt in range(max_attempts):
            x = margin + random.random() * (screen_width - margin * 2)
            y = margin + random.random() * (screen_height - margin * 2)
            
            # Check if position is valid
            too_close = any(math.hypot(s.x - x, s.y - y) < min_distance for s in game_state.stations)
            in_river = any(river.contains(x, y) for river in game_state.rivers)
            
            if not too_close and not in_river:
                # Determine station type
                station_type = self.get_new_station_type()
                game_state.stations.append(Station(x, y, station_type))
                return
        
        print("Warning: Could not find valid position for new station")
    
    def get_new_station_type(self):
        """Determine type for new station"""
        from state import game_state
        
        game_time = time.time() * 1000 - game_state.game_start_time
        minutes_played = game_time / 60000
        
        basic_types = STATION_TYPES.basic_types()
        special_types = STATION_TYPES.special_types()
        
        # After 2 minutes, 10% chance for special types
        if minutes_played > 2 and random.random() < 0.1:
            # Check for unused special types
            used_special = {s.type for s in game_state.stations if s.type in special_types}
            available_special = [t for t in special_types if t not in used_special]
            
            if available_special:
                return random.choice(available_special)
        
        return random.choice(basic_types)
    
    def check_game_over(self):
        """Check if game over conditions are met"""
        from state import game_state
        
        current_time = time.time() * 1000
        
        for station in game_state.stations:
            if station.overcrowd_start_time:
                if current_time - station.overcrowd_start_time > CONFIG.OVERCROWD_TIME:
                    return True
        
        return False
    
    def render(self, screen):
        """Render the game world"""
        from state import game_state
        from systems.input import InputHandler
        
        # Clear screen with background color
        screen.fill((244, 241, 233))
        
        if not self.initialized:
            return
        
        # Draw rivers
        for river in game_state.rivers:
            river.draw(screen)
        
        # Draw metro lines
        for line in game_state.lines:
            line.draw(screen)
        
        # Draw stations
        for station in game_state.stations:
            station.draw(screen)
        
        # Draw trains
        for train in game_state.trains:
            train.draw(screen)