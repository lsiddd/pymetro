# ./components/train.py
import math
from typing import List, Any
from config import CONFIG
from state import now_ms

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

        # Arc-length path for the current segment
        self._path_pts: List[Any] = []
        self._path_length: float = 0.0
        
        self.carriage_count: int = 0  # Number of carriages (0–MAX_CARRIAGES_PER_TRAIN)
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
    def has_carriage(self) -> bool:
        return self.carriage_count > 0

    @property
    def total_capacity(self) -> int:
        return self.capacity * (1 + self.carriage_count)
    
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

        # Use arc-length of the routed visual path, falling back to straight distance
        distance = self._path_length if self._path_length > 0 else math.hypot(
            next_station.x - current_station.x,
            next_station.y - current_station.y,
        )

        # Speed control with acceleration/deceleration
        # progress is in pixels traveled (0 → distance)
        accel_dist = min(distance * 0.4, 60)
        decel_start = max(distance - accel_dist, distance * 0.6)

        if self.progress < accel_dist:
            self.speed = min(self.max_speed, self.speed + CONFIG.TRAIN_ACCELERATION * delta_time)
        elif self.progress > decel_start:
            self.speed = max(0.2 * self.max_speed, self.speed - CONFIG.TRAIN_ACCELERATION * delta_time)
        else:
            self.speed = self.max_speed

        self.progress += self.speed * game_state.speed * (delta_time / 16.67)

        if self.progress >= distance:
            # Arrived at next station
            self.progress = distance
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
                game_state.carriages += self.carriage_count
                
                # Stop any further processing for this decommissioned train
                return
            # --- CHANGE END ---

            self.process_passengers(next_station)
            self.speed = 0
        else:
            # Interpolate position along the routed visual path
            if self._path_pts:
                self.x, self.y = self._get_pos_on_path(self.progress)
            else:
                dx = next_station.x - current_station.x
                dy = next_station.y - current_station.y
                d = math.hypot(dx, dy)
                if d > 0:
                    frac = self.progress / d
                    self.x = current_station.x + dx * frac
                    self.y = current_station.y + dy * frac
        

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

        self.progress = 0.0
        self._compute_path_waypoints()

    def _compute_path_waypoints(self) -> None:
        """Compute the visual waypoints for the current segment and store arc-length."""
        stations = self.line.stations
        if (self.current_station_index >= len(stations) or
                self.next_station_index >= len(stations) or
                self.next_station_index < 0):
            self._path_pts = []
            self._path_length = 0.0
            return

        s1 = stations[self.current_station_index]
        s2 = stations[self.next_station_index]
        try:
            self._path_pts = self.line.get_train_waypoints(s1, s2)
        except Exception:
            self._path_pts = [(s1.x, s1.y), (s2.x, s2.y)]

        total = 0.0
        for i in range(len(self._path_pts) - 1):
            a, b = self._path_pts[i], self._path_pts[i + 1]
            total += math.hypot(b[0] - a[0], b[1] - a[1])
        self._path_length = total if total > 0 else 1.0

    def _get_angle_on_path(self, dist: float) -> float:
        """Return the travel direction (radians) at arc-length *dist* along _path_pts."""
        pts = self._path_pts
        if not pts or len(pts) < 2:
            return 0.0
        remaining = max(0.0, dist)
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            if remaining <= seg_len or i == len(pts) - 2:
                return math.atan2(b[1] - a[1], b[0] - a[0])
            remaining -= seg_len
        # Fallback: direction of last segment
        return math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])

    def _get_pos_on_path(self, dist: float) -> tuple:
        """Return (x, y) at arc-length *dist* along self._path_pts."""
        pts = self._path_pts
        if not pts:
            return (self.x, self.y)
        remaining = max(0.0, dist)
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            if remaining <= seg_len or i == len(pts) - 2:
                t = (remaining / seg_len) if seg_len > 1e-6 else 1.0
                t = min(t, 1.0)
                return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            remaining -= seg_len
        return pts[-1]

    def process_passengers(self, station: Any) -> None:
        """Handle passenger boarding and alighting"""
        from state import game_state
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

        # Remove delivered passengers from the global list.
        # Transferred passengers have on_train=None (set in _should_alight_passenger);
        # delivered ones still point to this train.
        for p in alighted_passengers:
            if p.on_train is self:
                p.on_train = None
                if p in game_state.passengers:
                    game_state.passengers.remove(p)

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
            station.delivery_animation = {'start_time': now_ms(), 'duration': 500}
            return True

        if passenger.path and len(passenger.path) > passenger.path_index and passenger.path[passenger.path_index] == station:
            passenger.path_index += 1

            if passenger.path_index >= len(passenger.path): # Final destination
                game_state.score += 1
                game_state.passengers_delivered += 1
                station.delivery_animation = {'start_time': now_ms(), 'duration': 500}
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
    
