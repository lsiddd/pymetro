import pygame
import math
import time
from config import CONFIG, STATION_TYPES

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
                self.overcrowd_start_time = time.time() * 1000
        else:
            self.overcrowd_start_time = None
    
    def draw(self, screen):
        # Draw overcrowd animation first
        if self.overcrowd_start_time:
            elapsed = time.time() * 1000 - self.overcrowd_start_time
            progress = elapsed / CONFIG.OVERCROWD_TIME
            
            # Pulsing glow effect
            pulse = 10 + math.sin(time.time() * 1000 / 150) * 5
            glow_surface = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (211, 47, 47, 50), (25, 25), CONFIG.STATION_RADIUS + 8)
            screen.blit(glow_surface, (self.x - 25, self.y - 25))
            
            # Timer arc
            if progress < 1:
                angle = progress * 2 * math.pi
                pygame.draw.arc(screen, (211, 47, 47), 
                               (self.x - CONFIG.STATION_RADIUS - 8, self.y - CONFIG.STATION_RADIUS - 8,
                                2 * (CONFIG.STATION_RADIUS + 8), 2 * (CONFIG.STATION_RADIUS + 8)),
                               -math.pi/2, -math.pi/2 + angle, 4)
        
        # Draw station based on type
        self._draw_station_shape(screen)
        
        # Draw interchange indicator
        if self.is_interchange:
            pygame.draw.circle(screen, (51, 51, 51), (self.x, self.y), CONFIG.STATION_RADIUS + 8, 6)
            
            # Upgrade animation
            if self.animate_upgrade:
                elapsed = time.time() * 1000 - self.animate_upgrade['start_time']
                progress = elapsed / self.animate_upgrade['duration']
                
                if progress < 1:
                    pulse = 1 + 0.3 * math.sin(progress * math.pi * 4)
                    radius = int((CONFIG.STATION_RADIUS + 15) * pulse)
                    pygame.draw.circle(screen, (76, 175, 80), (self.x, self.y), radius, 4)
                else:
                    self.animate_upgrade = None
        
        # Draw delivery animation
        if self.delivery_animation:
            elapsed = time.time() * 1000 - self.delivery_animation['start_time']
            progress = elapsed / self.delivery_animation['duration']
            
            if progress < 1:
                scale = 1 + 0.5 * (1 - progress)
                alpha = int(255 * (1 - progress))
                radius = int(CONFIG.STATION_RADIUS * scale)
                
                # Create transparent surface for alpha blending
                glow_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (76, 175, 80, alpha), 
                                 (radius + 5, radius + 5), radius, 3)
                screen.blit(glow_surface, (self.x - radius - 5, self.y - radius - 5))
            else:
                self.delivery_animation = None
        
        # Draw passengers inside station
        self._draw_passengers(screen)
    
    def _draw_station_shape(self, screen):
        fill_color = (244, 241, 233)
        border_color = (51, 51, 51)
        radius = CONFIG.STATION_RADIUS
        
        if self.type == STATION_TYPES.CIRCLE:
            pygame.draw.circle(screen, fill_color, (self.x, self.y), radius)
            pygame.draw.circle(screen, border_color, (self.x, self.y), radius, 4)
        
        elif self.type == STATION_TYPES.TRIANGLE:
            h = radius * 1.7
            points = [
                (self.x, self.y - h * 0.6),
                (self.x - h * 0.5, self.y + h * 0.4),
                (self.x + h * 0.5, self.y + h * 0.4)
            ]
            pygame.draw.polygon(screen, fill_color, points)
            pygame.draw.polygon(screen, border_color, points, 4)
        
        elif self.type == STATION_TYPES.SQUARE:
            size = radius * 1.6
            rect = pygame.Rect(self.x - size/2, self.y - size/2, size, size)
            pygame.draw.rect(screen, fill_color, rect)
            pygame.draw.rect(screen, border_color, rect, 4)
        
        elif self.type == STATION_TYPES.PENTAGON:
            self._draw_polygon(screen, 5, radius * 1.4, fill_color, border_color)
        
        elif self.type == STATION_TYPES.DIAMOND:
            size = radius * 1.4
            points = [
                (self.x, self.y - size),
                (self.x + size, self.y),
                (self.x, self.y + size),
                (self.x - size, self.y)
            ]
            pygame.draw.polygon(screen, fill_color, points)
            pygame.draw.polygon(screen, border_color, points, 4)
        
        elif self.type == STATION_TYPES.STAR:
            self._draw_star(screen, radius * 1.5, fill_color, border_color)
        
        elif self.type == STATION_TYPES.CROSS:
            self._draw_cross(screen, radius * 1.4, fill_color, border_color)
    
    def _draw_polygon(self, screen, sides, radius, fill_color, border_color):
        points = []
        for i in range(sides):
            angle = (i * 2 * math.pi) / sides - math.pi / 2
            x = self.x + radius * math.cos(angle)
            y = self.y + radius * math.sin(angle)
            points.append((x, y))
        pygame.draw.polygon(screen, fill_color, points)
        pygame.draw.polygon(screen, border_color, points, 4)
    
    def _draw_star(self, screen, radius, fill_color, border_color):
        outer_radius = radius
        inner_radius = radius * 0.4
        spikes = 5
        points = []
        
        for i in range(spikes * 2):
            angle = (i * math.pi) / spikes - math.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            x = self.x + r * math.cos(angle)
            y = self.y + r * math.sin(angle)
            points.append((x, y))
        
        pygame.draw.polygon(screen, fill_color, points)
        pygame.draw.polygon(screen, border_color, points, 4)
    
    def _draw_cross(self, screen, size, fill_color, border_color):
        thickness = size * 0.3
        
        # Horizontal bar
        h_rect = pygame.Rect(self.x - size/2, self.y - thickness/2, size, thickness)
        pygame.draw.rect(screen, fill_color, h_rect)
        pygame.draw.rect(screen, border_color, h_rect, 4)
        
        # Vertical bar
        v_rect = pygame.Rect(self.x - thickness/2, self.y - size/2, thickness, size)
        pygame.draw.rect(screen, fill_color, v_rect)
        pygame.draw.rect(screen, border_color, v_rect, 4)
    
    def _draw_passengers(self, screen):
        passenger_area = CONFIG.STATION_RADIUS * 0.9
        cols = 3
        slot_size = CONFIG.PASSENGER_SIZE + 2
        start_x = self.x - (cols / 2 - 0.5) * slot_size
        start_y = self.y - passenger_area / 2 + CONFIG.PASSENGER_SIZE / 2
        
        for i, passenger in enumerate(self.passengers):
            if i < self.capacity:
                row = i // cols
                col = i % cols
                px = start_x + col * slot_size
                py = start_y + row * slot_size
                passenger.draw_at(screen, px, py)
            elif i == self.capacity:  # Show one extra to indicate overflow
                passenger.draw_at(screen, self.x, self.y + CONFIG.STATION_RADIUS + 15)