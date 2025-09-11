# ./components/train.py
import pygame
import math
import time
from typing import List, Optional, Set, Any
from config import CONFIG, STATION_TYPES

class Train:
    def __init__(self, line: Any):
        from state import game_state
        self.id: int = game_state.train_id_counter
        game_state.train_id_counter += 1
        
        self.line: Any = line
        self.passengers: List[Any] = []
        self.capacity: int = CONFIG.TRAIN_CAPACITY
        self.current_station_index: int = 0
        self.next_station_index: int = 1
        self.direction: int = 1
        self.x: float = 0
        self.y: float = 0
        self.progress: float = 0
        
        self.state: str = 'WAITING'  # WAITING, MOVING
        self.wait_timer: float = 0
        
        self.speed: float = 0
        self.max_speed: float = CONFIG.TRAIN_MAX_SPEED
        
        self.has_carriage: bool = False
        self.is_loop: bool = False
        
        if line.stations:
            self.x = line.stations[0].x
            self.y = line.stations[0].y
            self.process_passengers(line.stations[0])
        
        self.check_loop_status()
    
    def check_loop_status(self) -> None:
        """Check if the line forms a loop"""
        if len(self.line.stations) >= 3:
            self.is_loop = self.line.stations[0] == self.line.stations[-1]
        else:
            self.is_loop = False
    
    @property
    def total_capacity(self) -> int:
        """Total capacity including carriage"""
        return self.capacity + (CONFIG.TRAIN_CAPACITY if self.has_carriage else 0)
    
    def update(self, delta_time: float) -> None:
        """Update train movement and logic"""
        from state import game_state
        

        if not self.line.active or len(self.line.stations) < 2:
            return
        
        if self.state == 'WAITING':
            self.wait_timer -= delta_time
            if self.wait_timer <= 0:
                self.state = 'MOVING'
                self.determine_next_station()
            return
        
        self.check_loop_status()
        
        current_station = self.line.stations[self.current_station_index] if self.current_station_index < len(self.line.stations) else None
        next_station = self.line.stations[self.next_station_index] if self.next_station_index < len(self.line.stations) else None
        
        if not current_station or not next_station:
            self.state = 'WAITING'
            self.wait_timer = 100
            return
        
        # Calculate movement vector
        dx = next_station.x - current_station.x
        dy = next_station.y - current_station.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            # This can happen if a line has duplicate consecutive stations. Let's force progress.
            self.progress = 1
        
        # Speed control with acceleration/deceleration
        accel_dist = min(distance * 0.4, 60)
        decel_start = max(distance - accel_dist, distance * 0.6)
        current_pos = self.progress * distance
        
        if current_pos < accel_dist:
            self.speed = min(self.max_speed, self.speed + CONFIG.TRAIN_ACCELERATION * delta_time)
        elif current_pos > decel_start:
            self.speed = max(0.2 * self.max_speed, self.speed - CONFIG.TRAIN_ACCELERATION * delta_time)
        else:
            self.speed = self.max_speed
        
        # Update position
        if distance > 0: # Avoid division by zero
             self.progress += (self.speed * game_state.speed * (delta_time / 16.67)) / distance
        
        if self.progress >= 1:
            # Arrived at next station
            self.progress = 1
            self.x = next_station.x
            self.y = next_station.y
            self.current_station_index = self.next_station_index

            # --- CHANGE START ---
            # Graceful deletion logic
            is_at_terminal = not self.is_loop and (self.current_station_index == 0 or self.current_station_index >= len(self.line.stations) - 1)
            if self.line.marked_for_deletion and is_at_terminal:
                # Unload all remaining passengers
                for p in self.passengers:
                    p.on_train = None
                    p.start_transfer(next_station)
                    next_station.add_passenger(p)
                self.passengers.clear()
                
                # Remove self from the game and return resources
                if self in game_state.trains:
                    game_state.trains.remove(self)
                if self in self.line.trains:
                    self.line.trains.remove(self)
                
                game_state.available_trains += 1
                if self.has_carriage:
                    game_state.carriages += 1
                
                # Stop any further processing for this decommissioned train
                return
            # --- CHANGE END ---

            self.process_passengers(next_station)
            self.speed = 0
        else:
            # Interpolate position
            self.x = current_station.x + dx * self.progress
            self.y = current_station.y + dy * self.progress
        

    def determine_next_station(self) -> None:
        """Determine next station to travel to"""
        if self.current_station_index >= len(self.line.stations):
            self.current_station_index = 0
        
        if self.is_loop:
            self.next_station_index = (self.current_station_index + 1) % (len(self.line.stations) - 1)
        else:
            if self.current_station_index >= len(self.line.stations) - 1:
                self.direction = -1
            elif self.current_station_index == 0:
                self.direction = 1
            
            self.next_station_index = self.current_station_index + self.direction
        
        self.progress = 0

    def process_passengers(self, station: Any) -> None:
        """Handle passenger boarding and alighting"""
        passengers_changed = 0
        
        # Alight passengers
        alighted_passengers = []
        remaining_passengers = []
        for p in self.passengers:
            if self._should_alight_passenger(p, station):
                alighted_passengers.append(p)
                passengers_changed +=1
            else:
                remaining_passengers.append(p)
        self.passengers = remaining_passengers

        # Board new passengers
        available_space = self.total_capacity - len(self.passengers)
        # --- CHANGE START ---
        # Prevent boarding if the line is being deleted
        if available_space > 0 and not self.line.marked_for_deletion:
        # --- CHANGE END ---
            upcoming_stops = self.get_upcoming_stops(station, single_direction_only=True)
            
            for passenger in station.passengers:
                passenger.update_wait_time(station)
            
            waiting_passengers = [p for p in station.passengers if self._can_board_passenger(p, upcoming_stops)]
            
            to_board = waiting_passengers[:available_space]
            for passenger in to_board:
                passengers_changed += 1
                station.remove_passenger(passenger)
                self.passengers.append(passenger)
                passenger.on_train = self
                passenger.current_station = None
        
        # Set wait time
        self.state = 'WAITING'
        base_time = CONFIG.INTERCHANGE_TRANSFER_TIME if station.is_interchange else CONFIG.PASSENGER_BOARD_TIME
        self.wait_timer = passengers_changed * base_time
        if self.wait_timer <= 0:
            self.wait_timer = 300

    def _should_alight_passenger(self, passenger: Any, station: Any) -> bool:
        """Check if passenger should get off at this station"""
        from state import game_state
        
        if passenger.destination == station.type:
            game_state.score += 1
            game_state.passengers_delivered += 1
            station.delivery_animation = {'start_time': time.time() * 1000, 'duration': 500}
            return True
        
        if passenger.path and len(passenger.path) > passenger.path_index and passenger.path[passenger.path_index] == station:
            passenger.path_index += 1
            
            if passenger.path_index >= len(passenger.path): # Final destination
                game_state.score += 1
                game_state.passengers_delivered += 1
                station.delivery_animation = {'start_time': time.time() * 1000, 'duration': 500}
                return True
            
            # Check for transfer
            next_stop_in_path = passenger.path[passenger.path_index]
            if next_stop_in_path not in self.get_upcoming_stops(station, True):
                passenger.on_train = None
                passenger.start_transfer(station)
                station.add_passenger(passenger)
                return True
        return False
    
    def _can_board_passenger(self, passenger: Any, upcoming_stops: List[Any]) -> bool:
        """Check if passenger can board this train"""
        if not passenger.path or passenger.path_index >= len(passenger.path):
            passenger.recalculate_path()
            if not passenger.path: return False
        
        return passenger.path[passenger.path_index] in upcoming_stops
    
    def get_upcoming_stops(self, current_station: Any, single_direction_only: bool = False) -> List[Any]:
        """Get list of upcoming stops on this train's route"""
        # (This function can remain the same as the original)
        upcoming = set()
        line_stations = self.line.stations
        num_stations = len(line_stations) - 1 if self.is_loop else len(line_stations)
        
        if num_stations <= 1: return []
        
        try:
            current_index = line_stations.index(current_station)
        except ValueError:
            # This can happen if the line was modified while the train was moving
            # Find the closest station on the line and snap to it
            min_dist = float('inf')
            closest_idx = -1
            for i, station in enumerate(line_stations):
                dist = math.hypot(self.x - station.x, self.y - station.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            current_index = closest_idx if closest_idx != -1 else 0


        current_direction = self.direction
        if not self.is_loop:
            if current_index == 0: current_direction = 1
            if current_index >= len(line_stations) - 1: current_direction = -1
        
        if self.is_loop:
            for i in range(1, num_stations):
                index = (current_index + i) % num_stations
                upcoming.add(line_stations[index])
        else:
            if current_direction == 1:
                for i in range(current_index + 1, len(line_stations)): upcoming.add(line_stations[i])
            else:
                for i in range(current_index - 1, -1, -1): upcoming.add(line_stations[i])
            
            if not single_direction_only:
                if current_direction == 1:
                    for i in range(len(line_stations) - 2, -1, -1): upcoming.add(line_stations[i])
                else:
                    for i in range(1, len(line_stations)): upcoming.add(line_stations[i])
        
        return list(upcoming)
    
    def reassign_to_line(self, new_line: Any) -> None:
        """Reassign train to a different line"""
        if self.line and self in self.line.trains: self.line.trains.remove(self)
        self.line = new_line
        new_line.trains.append(self)
        self.current_station_index = 0
        self.next_station_index = 1 if len(new_line.stations) > 1 else 0
        self.direction = 1
        self.progress = 0
        self.state = 'WAITING'
        self.wait_timer = 500
        if new_line.stations: self.x, self.y = new_line.stations[0].x, new_line.stations[0].y
        self.check_loop_status()
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the train on screen"""
        # (This function can remain the same as the original, but with print statements removed)
        if not self.line.active or len(self.line.stations) < 2: return
            
        width, height = CONFIG.TRAIN_WIDTH, CONFIG.TRAIN_HEIGHT
        
        current_st = self.line.stations[self.current_station_index] if self.current_station_index < len(self.line.stations) else None
        next_st = self.line.stations[self.next_station_index] if self.next_station_index < len(self.line.stations) else None
        
        if not current_st: return
        next_st = next_st or current_st
        
        angle: float = 0.0
        if next_st != current_st:
            angle = math.atan2(next_st.y - current_st.y, next_st.x - current_st.x)
        
        total_width = width + (width + 5 if self.has_carriage else 0)
        train_surface = pygame.Surface((total_width, height), pygame.SRCALPHA)
        
        pygame.draw.rect(train_surface, self.line.color, (0, 0, width, height))
        pygame.draw.rect(train_surface, (51, 51, 51), (0, 0, width, height), 2)
        
        if self.has_carriage:
            pygame.draw.rect(train_surface, self.line.color, (width + 5, 0, width, height))
            pygame.draw.rect(train_surface, (51, 51, 51), (width + 5, 0, width, height), 2)
            pygame.draw.line(train_surface, (51, 51, 51), (width, height//2), (width + 5, height//2), 1)
        
        if self.passengers:
            max_dots = min(len(self.passengers), 4)
            for i in range(max_dots):
                dot_x = width//4 + (i % 2) * width//4
                dot_y = height//4 + (i // 2) * height//4
                pygame.draw.circle(train_surface, (255, 255, 255), (dot_x, dot_y), 2)
        
        rotated_surface = pygame.transform.rotate(train_surface, -math.degrees(angle))
        rotated_rect = rotated_surface.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated_surface, rotated_rect)