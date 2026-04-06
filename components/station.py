from config import CONFIG
from state import now_ms

class Station:
    def __init__(self, x, y, station_type):
        from state import game_state
        self.id = game_state.station_id_counter
        game_state.station_id_counter += 1
        
        self.x = x
        self.y = y
        self.type = station_type
        self.passengers = []
        self.is_interchange = False
        self.overcrowd_start_time = None
        
        # Animation properties
        self.connection_animation = None
        self.delivery_animation = None
        self.animate_upgrade = None
    
    @property
    def capacity(self):
        return CONFIG.MAX_PASSENGERS_WITH_INTERCHANGE if self.is_interchange else CONFIG.MAX_PASSENGERS_PER_STATION
    
    def add_passenger(self, passenger):
        if len(self.passengers) < 100:  # Absolute max
            self.passengers.append(passenger)
            self.check_overcrowd()
    
    def remove_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)
        self.check_overcrowd()
    
    def check_overcrowd(self):
        if len(self.passengers) > self.capacity:
            if not self.overcrowd_start_time:
                self.overcrowd_start_time = now_ms()
        else:
            self.overcrowd_start_time = None
    
