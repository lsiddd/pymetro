# ./components/line.py
import pygame
import math
import time
from typing import List, Optional, Tuple, Dict, Any, Union
from config import CONFIG

class Line:
    def __init__(self, color: str, index: int):
        self.color: Tuple[int, int, int] = self._hex_to_rgb(color)
        self.index: int = index
        self.stations: List[Any] = []
        self.trains: List[Any] = []
        self.active: bool = False
        # --- CHANGE START ---
        self.marked_for_deletion: bool = False
        # --- CHANGE END ---
        self.original_start: Optional[Any] = None
        self.original_end: Optional[Any] = None
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        rgb_values = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb_values[0], rgb_values[1], rgb_values[2])
    
    def add_station(self, station: Any, insert_index: int = -1) -> bool:
        """Add a station to this line"""
        if len(self.stations) == 0:
            self.original_start = station
        
        # --- BUG FIX ---
        # More robust check to prevent adding a station that is already an endpoint,
        # which would create a zero-length segment.
        if len(self.stations) > 0:
            if insert_index == 0 and station == self.stations[0]:
                return False # Trying to add to start, but it's already the start
            is_append = insert_index == -1 or insert_index >= len(self.stations)
            if is_append and station == self.stations[-1]:
                return False # Trying to add to end, but it's already the end
        # --- END FIX ---

        is_closing_loop = (
            len(self.stations) >= 2 and
            station == self.stations[0] and
            (insert_index == -1 or insert_index == len(self.stations))
        )
        
        if station in self.stations and not is_closing_loop:
            return False
        
        if insert_index == 0:
            self.stations.insert(0, station)
            self.original_start = station
        elif 0 < insert_index < len(self.stations):
            self.stations.insert(insert_index, station)
        else:
            self.stations.append(station)
            if not is_closing_loop:
                self.original_end = station
        
        self.active = len(self.stations) >= 2
        self._animate_station_connection(station)
        
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
        
        return True
    
    def remove_end_station(self, station: Any) -> None:
        """Remove a station from the ends of the line"""
        if len(self.stations) < 2:
            return
        
        is_loop = len(self.stations) > 2 and self.stations[0] == self.stations[-1]
        
        try:
            index = self.stations.index(station)
            last_index = len(self.stations) - 1
            
            if is_loop:
                if station == self.original_start and (index == 0 or index == last_index):
                    self.stations.pop()
                    self.original_end = self.stations[-1] if self.stations else None
                elif station == self.original_end:
                    self.stations.pop()
                    self.stations.reverse()
                    self.original_start = self.stations[0] if self.stations else None
                    self.original_end = self.stations[-1] if self.stations else None
            else:
                if index == 0:
                    self.stations.pop(0)
                    self.original_start = self.stations[0] if self.stations else None
                elif index == last_index:
                    self.stations.pop()
                    self.original_end = self.stations[-1] if self.stations else None
        except ValueError:
            pass  # Station not in list
        
        # --- CHANGE START ---
        if len(self.stations) < 2:
            self.marked_for_deletion = True
        else:
            self.active = True
        # --- CHANGE END ---
            
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
    
    def remove_station(self, station: Any) -> None:
        """Remove a station from anywhere in the line"""
        indices = [i for i, s in enumerate(self.stations) if s == station]
        
        if not indices:
            return
        
        is_loop = len(self.stations) > 2 and self.stations[0] == self.stations[-1]
        is_endpoint = not is_loop and (indices[0] == 0 or indices[0] == len(self.stations) - 1)
        
        if is_endpoint:
            self.remove_end_station(station)
            return
        
        if len(indices) > 1:  # Station appears multiple times (loop case)
            self.stations.pop()  # Remove duplicate at end
            self.stations.pop(indices[0])  # Remove from middle
            if self.stations:
                self.stations.append(self.stations[0])  # Re-close loop
        else:
            self.stations.pop(indices[0])
        
        self.active = len(self.stations) >= 2
        # --- CHANGE START ---
        if not self.active:
            self.marked_for_deletion = True
        # --- CHANGE END ---
            
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
    
    def clear_line(self) -> None:
        """Clear the entire line and return resources"""
        from state import game_state
        
        # Calculate refunded bridges
        refunded_bridges = 0
        for i in range(len(self.stations) - 1):
            if self.check_river_crossing(self.stations[i], self.stations[i+1]):
                refunded_bridges += 1
        game_state.bridges += refunded_bridges
        
        # Return trains to available pool
        for train in self.trains[:]:
            if train in game_state.trains:
                game_state.trains.remove(train)
            game_state.available_trains += 1
        
        # Reset line state
        self.stations = []
        self.trains = []
        self.active = False
        # --- CHANGE START ---
        self.marked_for_deletion = False
        # --- CHANGE END ---
        self.original_start = None
        self.original_end = None
        
        # Mark pathfinding graph as dirty
        from systems.pathfinding import mark_graph_dirty
        mark_graph_dirty()
    
    def check_river_crossing(self, station1: Any, station2: Any) -> bool:
        """Check if line segment crosses a river"""
        from state import game_state
        
        if not station1 or not station2:
            return False
        
        for river in game_state.rivers:
            if self._line_intersects_polygon(
                station1.x, station1.y,
                station2.x, station2.y,
                river.points
            ):
                return True
        return False
    
    def _line_intersects_polygon(self, x1: float, y1: float, x2: float, y2: float, polygon_points: List[Any]) -> bool:
        """Check if line intersects with polygon"""
        try:
            for i in range(len(polygon_points)):
                p1 = polygon_points[i]
                p2 = polygon_points[(i + 1) % len(polygon_points)]
                
                # Handle both dict and object formats
                p1x = p1['x'] if isinstance(p1, dict) else p1.x
                p1y = p1['y'] if isinstance(p1, dict) else p1.y
                p2x = p2['x'] if isinstance(p2, dict) else p2.x
                p2y = p2['y'] if isinstance(p2, dict) else p2.y
                
                if self._line_intersects_line(x1, y1, x2, y2, p1x, p1y, p2x, p2y):
                    return True
            return False
        except (KeyError, AttributeError, TypeError) as e:
            print(f"Line: Error checking polygon intersection: {e}")
            return False
    
    def _line_intersects_line(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float) -> bool:
        """Check if two line segments intersect"""
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0:
            return False
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        return 0 < t < 1 and 0 < u < 1
    
    def _animate_station_connection(self, station: Any) -> None:
        """Animate station connection effect"""
        if station:
            station.connection_animation = {
                'start_time': time.time() * 1000,
                'duration': 300
            }
    
    def find_insertion_point(self, station: Any) -> int:
        """Find best insertion point for a station in the line"""
        if len(self.stations) < 2:
            return -1
        
        min_distance = float('inf')
        insert_index = -1
        
        for i in range(len(self.stations) - 1):
            s1 = self.stations[i]
            s2 = self.stations[i + 1]
            distance = self._distance_to_line_segment(station, s1, s2)
            
            if distance < min_distance and distance < 50:
                min_distance = distance
                insert_index = i + 1
        
        return insert_index
    
    def _distance_to_line_segment(self, point: Any, line_start: Any, line_end: Any) -> float:
        """Calculate distance from point to line segment"""
        A = point.x - line_start.x
        B = point.y - line_start.y
        C = line_end.x - line_start.x
        D = line_end.y - line_start.y
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
        
        if param < 0:
            xx, yy = line_start.x, line_start.y
        elif param > 1:
            xx, yy = line_end.x, line_end.y
        else:
            xx = line_start.x + param * C
            yy = line_start.y + param * D
        
        dx = point.x - xx
        dy = point.y - yy
        return math.sqrt(dx * dx + dy * dy)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draw the line on screen"""
        # --- CHANGE START ---
        if (len(self.stations) < 2 and not self.marked_for_deletion) or not self.active:
            return
        # --- CHANGE END ---
        
        from state import game_state
        
        for i in range(len(self.stations) - 1):
            s1 = self.stations[i]
            s2 = self.stations[i + 1]
            
            # Calculate offset for multiple lines between same stations
            shared_lines = [line for line in game_state.lines
                           if line.active and s1 in line.stations and s2 in line.stations
                           and abs(line.stations.index(s1) - line.stations.index(s2)) == 1]
            
            line_index = next((i for i, line in enumerate(shared_lines) if line.index == self.index), 0)
            offset = (line_index - (len(shared_lines) - 1) / 2) * (CONFIG.LINE_WIDTH + 2)
            
            # Calculate perpendicular offset
            dx = s2.x - s1.x
            dy = s2.y - s1.y
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                nx = -dy / length
                ny = dx / length
                
                start_x = s1.x + offset * nx
                start_y = s1.y + offset * ny
                end_x = s2.x + offset * nx
                end_y = s2.y + offset * ny
                
                # Check if crosses river
                crosses_river = self.check_river_crossing(s1, s2)
                
                # --- CHANGE START ---
                # Use a faded color if marked for deletion
                line_color = self.color
                if self.marked_for_deletion:
                    line_color = (
                        int(self.color[0] * 0.5 + 128 * 0.5),
                        int(self.color[1] * 0.5 + 128 * 0.5),
                        int(self.color[2] * 0.5 + 128 * 0.5)
                    )

                if crosses_river:
                    # Draw dashed line for bridge
                    self._draw_dashed_line(screen, start_x, start_y, end_x, end_y, 
                                         line_color, int(CONFIG.LINE_WIDTH * 0.8))
                elif self.marked_for_deletion:
                    # Always draw marked lines as dashed
                    self._draw_dashed_line(screen, start_x, start_y, end_x, end_y,
                                         line_color, CONFIG.LINE_WIDTH)
                else:
                    # Draw solid line
                    pygame.draw.line(screen, line_color, 
                                   (start_x, start_y), (end_x, end_y), CONFIG.LINE_WIDTH)
                # --- CHANGE END ---
    
    def _draw_dashed_line(self, screen: pygame.Surface, x1: float, y1: float, x2: float, y2: float, color: Tuple[int, int, int], width: int) -> None:
        """Draw a dashed line"""
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            return
        
        dash_length = 8
        gap_length = 4
        total_length = dash_length + gap_length
        
        num_dashes = int(distance / total_length)
        
        for i in range(num_dashes + 1):
            start_ratio = (i * total_length) / distance
            end_ratio = min((i * total_length + dash_length) / distance, 1.0)
            
            if start_ratio >= 1.0:
                break
            
            start_x = x1 + dx * start_ratio
            start_y = y1 + dy * start_ratio
            end_x = x1 + dx * end_ratio
            end_y = y1 + dy * end_ratio
            
            pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), width)