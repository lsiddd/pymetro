from config import CONFIG
from state import now_ms

class Passenger:
    def __init__(self, station, destination):
        self.current_station = station
        self.destination = destination
        self.on_train = None
        self.path = None
        self.path_index = 0
        self.wait_start_time = now_ms()
        self.last_route_calculation = self.wait_start_time
        self.recalculate_path()
    
    def recalculate_path(self):
        """Calculate and store the path to the destination"""
        from systems.pathfinding import find_path
        self.path = find_path(self.current_station, self.destination)
        self.path_index = 1 if self.path else 0  # Start by targeting second station in path
        self.last_route_calculation = now_ms()
    
    def update_wait_time(self, current_station):
        """Update wait tracking and potentially recalculate route"""
        if self.current_station != current_station:
            # Passenger has moved to a new station
            self.current_station = current_station
            self.wait_start_time = now_ms()
            return

        current_time = now_ms()
        wait_duration = current_time - self.wait_start_time
        
        # If passenger has been waiting too long, consider re-routing
        if (wait_duration > CONFIG.PASSENGER_WAIT_PATIENCE and 
            current_time - self.last_route_calculation > CONFIG.PASSENGER_WAIT_PATIENCE):
            self.recalculate_path()
    
    def start_transfer(self, station):
        """Called when passenger transfers to a new station"""
        self.current_station = station
        self.wait_start_time = now_ms()
        # Use slower transfer time for regular stations
        if not station.is_interchange:
            self.wait_start_time += CONFIG.REGULAR_TRANSFER_TIME - CONFIG.INTERCHANGE_TRANSFER_TIME
    
