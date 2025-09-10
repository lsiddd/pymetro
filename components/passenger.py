import pygame
import math
import time
from config import CONFIG, STATION_TYPES

class Passenger:
    def __init__(self, station, destination):
        self.current_station = station
        self.destination = destination
        self.on_train = None
        self.path = None
        self.path_index = 0
        self.wait_start_time = time.time() * 1000
        self.last_route_calculation = self.wait_start_time
        self.recalculate_path()
    
    def recalculate_path(self):
        """Calculate and store the path to the destination"""
        from systems.pathfinding import find_path
        self.path = find_path(self.current_station, self.destination)
        self.path_index = 1 if self.path else 0  # Start by targeting second station in path
        self.last_route_calculation = time.time() * 1000
    
    def update_wait_time(self, current_station):
        """Update wait tracking and potentially recalculate route"""
        if self.current_station != current_station:
            # Passenger has moved to a new station
            self.current_station = current_station
            self.wait_start_time = time.time() * 1000
            return
        
        current_time = time.time() * 1000
        wait_duration = current_time - self.wait_start_time
        
        # If passenger has been waiting too long, consider re-routing
        if (wait_duration > CONFIG.PASSENGER_WAIT_PATIENCE and 
            current_time - self.last_route_calculation > CONFIG.PASSENGER_WAIT_PATIENCE):
            self.recalculate_path()
    
    def start_transfer(self, station):
        """Called when passenger transfers to a new station"""
        self.current_station = station
        self.wait_start_time = time.time() * 1000
        # Use slower transfer time for regular stations
        if not station.is_interchange:
            self.wait_start_time += CONFIG.REGULAR_TRANSFER_TIME - CONFIG.INTERCHANGE_TRANSFER_TIME
    
    def draw_at(self, screen, x, y):
        """Draw passenger at specific coordinates"""
        color = (51, 51, 51)
        size = CONFIG.PASSENGER_SIZE
        
        if self.destination == STATION_TYPES.CIRCLE:
            pygame.draw.circle(screen, color, (int(x), int(y)), size//2)
        
        elif self.destination == STATION_TYPES.TRIANGLE:
            points = [
                (x, y - size//2),
                (x - size//2, y + size//2),
                (x + size//2, y + size//2)
            ]
            pygame.draw.polygon(screen, color, points)
        
        elif self.destination == STATION_TYPES.SQUARE:
            rect = pygame.Rect(x - size//2, y - size//2, size, size)
            pygame.draw.rect(screen, color, rect)
        
        elif self.destination == STATION_TYPES.PENTAGON:
            self._draw_passenger_polygon(screen, x, y, 5, size//2, color)
        
        elif self.destination == STATION_TYPES.DIAMOND:
            points = [
                (x, y - size//2),
                (x + size//2, y),
                (x, y + size//2),
                (x - size//2, y)
            ]
            pygame.draw.polygon(screen, color, points)
        
        elif self.destination == STATION_TYPES.STAR:
            self._draw_passenger_star(screen, x, y, size//2, color)
        
        elif self.destination == STATION_TYPES.CROSS:
            thickness = size * 0.3
            # Horizontal bar
            h_rect = pygame.Rect(x - size//2, y - thickness//2, size, thickness)
            pygame.draw.rect(screen, color, h_rect)
            # Vertical bar
            v_rect = pygame.Rect(x - thickness//2, y - size//2, thickness, size)
            pygame.draw.rect(screen, color, v_rect)
    
    def _draw_passenger_polygon(self, screen, x, y, sides, radius, color):
        points = []
        for i in range(sides):
            angle = (i * 2 * math.pi) / sides - math.pi / 2
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
        pygame.draw.polygon(screen, color, points)
    
    def _draw_passenger_star(self, screen, x, y, radius, color):
        outer_radius = radius
        inner_radius = radius * 0.4
        spikes = 5
        points = []
        
        for i in range(spikes * 2):
            angle = (i * math.pi) / spikes - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            px = x + r * math.cos(angle)
            py = y + r * math.sin(angle)
            points.append((px, py))
        
        pygame.draw.polygon(screen, color, points)