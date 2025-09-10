import pygame
import math
import time
from config import CONFIG

class InputHandler:
    def __init__(self):
        # Line drawing state (matching JS version exactly)
        self.is_drawing_line = False
        self.is_removing_segment = False
        self.dragged_segment = None
        self.current_path = []
        
        # Resource dragging
        self.dragged_train_resource = False
        self.dragged_carriage = False
        
        # Hover state
        self.hovered_station = None
        self.hovered_train = None
        
        # Preview line (matching JS version)
        self.preview_line = None
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_pressed = False
        
        print("InputHandler: Initialized with JS-style line drawing")
    
    def update(self, mouse_pos, mouse_pressed):
        """Update input state - called every frame"""
        self.mouse_pos = mouse_pos
        self.mouse_pressed = mouse_pressed
        
        # Update hovered station
        self._update_hovered_station()
        
        # Update preview line for smooth animation
        self._update_preview_line()
    
    def _update_hovered_station(self):
        """Update which station is being hovered"""
        from state import game_state
        
        self.hovered_station = None
        for station in game_state.stations:
            distance = math.sqrt((self.mouse_pos[0] - station.x)**2 + (self.mouse_pos[1] - station.y)**2)
            if distance <= CONFIG.STATION_RADIUS + 15:  # Match JS version threshold
                self.hovered_station = station
                break
    
    def _update_preview_line(self):
        """Update line drawing preview - matches JS version"""
        from state import game_state
        
        if self.is_drawing_line and len(self.current_path) > 0:
            start_station = self.current_path[0]
            
            if self.is_removing_segment:
                # Check if we've moved far enough to switch from remove to extend mode
                distance = math.sqrt((self.mouse_pos[0] - start_station.x)**2 + (self.mouse_pos[1] - start_station.y)**2)
                if distance > CONFIG.STATION_RADIUS * 2.5:
                    self.is_removing_segment = False
            
            # Set preview line (null if removing, line if extending)
            if self.is_removing_segment:
                self.preview_line = None
            else:
                self.preview_line = {
                    'start': {'x': start_station.x, 'y': start_station.y},
                    'end': {'x': self.mouse_pos[0], 'y': self.mouse_pos[1]},
                    'color': game_state.lines[game_state.selected_line].color,
                    'is_segment': False
                }
        
        elif self.dragged_segment:
            # Preview for segment insertion
            s1 = self.dragged_segment['line'].stations[self.dragged_segment['index']]
            s2 = self.dragged_segment['line'].stations[self.dragged_segment['index'] + 1]
            
            # Find target (station or mouse pos)
            target = self.hovered_station if self.hovered_station else {'x': self.mouse_pos[0], 'y': self.mouse_pos[1]}
            
            self.preview_line = {
                'is_segment': True,
                's1': s1,
                's2': s2, 
                'target': target,
                'color': self.dragged_segment['line'].color
            }
        
        else:
            self.preview_line = None
    
    def handle_mouse_down(self, pos, button):
        """Handle mouse button down - matches JS version logic"""
        from state import game_state
        from components.train import Train
        from systems.ui import update_resource_display
        
        if game_state.game_over:
            return False
        
        if button == 1:  # Left click
            print(f"InputHandler: Mouse down at {pos}")
            
            line = game_state.lines[game_state.selected_line]
            
            # Check if clicking on a station
            station = self._get_station_at_pos(pos)
            if station:
                print(f"InputHandler: Clicked on station at ({station.x}, {station.y})")
                
                # Handle shift+click for removal
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    if station in line.stations:
                        print(f"InputHandler: Removing station from line {game_state.selected_line}")
                        line.remove_station(station)
                        return True
                
                # Check if this is a loop closure attempt
                is_already_loop = len(line.stations) > 2 and line.stations[0] == line.stations[-1]
                if is_already_loop and station in line.stations:
                    print("InputHandler: Ignoring click on station in completed loop")
                    return False
                
                # Start line drawing
                self.is_drawing_line = True
                
                if station in line.stations:
                    # Clicking on existing station - check if it's an endpoint
                    station_index = None
                    try:
                        station_index = line.stations.index(station)
                    except ValueError:
                        pass
                    
                    if station_index == 0 or station_index == len(line.stations) - 1:
                        # Endpoint - start removal mode initially
                        self.current_path = [station]
                        self.is_removing_segment = True
                        print(f"InputHandler: Started removal mode from endpoint")
                    else:
                        # Middle station - can't extend from here
                        self.is_drawing_line = False
                        print(f"InputHandler: Can't extend from middle station")
                else:
                    # New station
                    self.current_path = [station]
                    self.is_removing_segment = False
                    print(f"InputHandler: Started line drawing from new station")
                
                return True
            
            # Check if clicking on a line segment for insertion
            for line_obj in game_state.lines:
                if not line_obj.active:
                    continue
                for i in range(len(line_obj.stations) - 1):
                    s1 = line_obj.stations[i]
                    s2 = line_obj.stations[i + 1]
                    if line_obj._distance_to_line_segment(type('pos', (), {'x': pos[0], 'y': pos[1]}), s1, s2) < 20:
                        self.dragged_segment = {'line': line_obj, 'index': i}
                        self.is_drawing_line = False
                        print(f"InputHandler: Started segment dragging on line {line_obj.index}")
                        return True
        
        elif button == 3:  # Right click - remove station from lines
            station = self._get_station_at_pos(pos)
            if station:
                print(f"InputHandler: Right-click removal of station")
                for line in game_state.lines:
                    if station in line.stations:
                        line.remove_station(station)
                return True
        
        return False
    
    def handle_mouse_up(self, pos, button):
        """Handle mouse button up - matches JS version logic"""
        from state import game_state
        from components.train import Train
        from systems.ui import update_resource_display
        
        if button != 1:
            return False
        
        print(f"InputHandler: Mouse up at {pos}")
        
        line = game_state.lines[game_state.selected_line]
        target_station = self._get_station_at_pos(pos)
        
        # Handle segment removal
        if self.is_removing_segment and len(self.current_path) > 0:
            start_station = self.current_path[0]
            
            # If we didn't drag to a new station, it's a removal action
            if not target_station or target_station == start_station:
                print(f"InputHandler: Finalizing station removal")
                
                # Find neighbor for bridge refund
                if line.stations:
                    s1 = line.stations[0] if len(line.stations) > 0 else None
                    if s1 and start_station == s1 and len(line.stations) > 1:
                        neighbor = line.stations[1]
                    elif len(line.stations) > 1:
                        neighbor = line.stations[-2] if start_station == line.stations[-1] else None
                    else:
                        neighbor = None
                    
                    # Check for bridge refund
                    if neighbor and line.check_river_crossing(start_station, neighbor):
                        game_state.bridges += 1
                        print(f"InputHandler: Refunded bridge, now have {game_state.bridges}")
                
                line.remove_end_station(start_station)
        
        # Handle segment insertion
        elif self.dragged_segment and target_station:
            segment = self.dragged_segment
            line_obj = segment['line']
            index = segment['index']
            
            if target_station not in line_obj.stations:
                print(f"InputHandler: Inserting station into line segment")
                
                s1 = line_obj.stations[index]
                s2 = line_obj.stations[index + 1]
                
                # Calculate bridge costs
                needs_bridge1 = line_obj.check_river_crossing(s1, target_station)
                needs_bridge2 = line_obj.check_river_crossing(target_station, s2)
                had_bridge = line_obj.check_river_crossing(s1, s2)
                bridge_cost = (1 if needs_bridge1 else 0) + (1 if needs_bridge2 else 0) - (1 if had_bridge else 0)
                
                if game_state.bridges >= bridge_cost:
                    game_state.bridges -= bridge_cost
                    line_obj.add_station(target_station, index + 1)
                    print(f"InputHandler: Successfully inserted station, bridges: {game_state.bridges}")
                else:
                    print(f"InputHandler: Not enough bridges for insertion (need {bridge_cost})")
        
        # Handle line extension/creation
        elif self.is_drawing_line and len(self.current_path) > 0 and target_station:
            start_station = self.current_path[0]
            
            if target_station != start_station:
                print(f"InputHandler: Extending/creating line from {start_station} to {target_station}")
                
                stations_before = len(line.stations)
                needs_bridge = line.check_river_crossing(start_station, target_station)
                
                if needs_bridge and game_state.bridges <= 0:
                    print(f"InputHandler: Not enough bridges for connection")
                else:
                    success = False
                    
                    # Case 1: Empty line
                    if len(line.stations) == 0:
                        print(f"InputHandler: Creating new line")
                        line.add_station(start_station)
                        success = line.add_station(target_station)
                    
                    # Case 2: Extend from start
                    elif len(line.stations) > 0 and line.stations[0] == start_station:
                        print(f"InputHandler: Extending from line start")
                        success = line.add_station(target_station, 0)
                    
                    # Case 3: Extend from end
                    elif len(line.stations) > 0 and line.stations[-1] == start_station:
                        print(f"InputHandler: Extending from line end")
                        success = line.add_station(target_station)
                    
                    # Case 4: Connect to start
                    elif len(line.stations) > 0 and line.stations[0] == target_station:
                        print(f"InputHandler: Connecting to line start")
                        success = line.add_station(start_station, 0)
                    
                    # Case 5: Connect to end
                    elif len(line.stations) > 0 and line.stations[-1] == target_station:
                        print(f"InputHandler: Connecting to line end")
                        success = line.add_station(start_station)
                    
                    if success:
                        print(f"InputHandler: Line operation successful")
                        
                        if needs_bridge:
                            game_state.bridges -= 1
                            print(f"InputHandler: Used bridge, now have {game_state.bridges}")
                        
                        # Add train if line was just created
                        if stations_before < 2 and len(line.stations) >= 2:
                            if game_state.available_trains > 0 and len(line.trains) == 0:
                                print(f"InputHandler: Adding train to new line")
                                game_state.available_trains -= 1
                                new_train = Train(line)
                                game_state.trains.append(new_train)
                                line.trains.append(new_train)
                    else:
                        print(f"InputHandler: Line operation failed")
        
        # Reset state
        self._reset_input_state()
        return True
    
    def handle_mouse_motion(self, pos):
        """Handle mouse motion"""
        # Mouse position is updated in update() method
        pass
    
    def _reset_input_state(self):
        """Reset all input state"""
        self.is_drawing_line = False
        self.is_removing_segment = False
        self.dragged_segment = None
        self.current_path = []
        self.preview_line = None
        
        print("InputHandler: Reset input state")
    
    def _get_station_at_pos(self, pos):
        """Get station at position"""
        from state import game_state
        
        for station in game_state.stations:
            distance = math.sqrt((pos[0] - station.x)**2 + (pos[1] - station.y)**2)
            if distance <= CONFIG.STATION_RADIUS + 15:
                return station
        return None
    
    def draw_preview(self, screen):
        """Draw line preview with smooth animation"""
        if not self.preview_line:
            return
        
        color = self.preview_line['color']
        
        # Create surface with alpha for smooth preview
        preview_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        alpha_color = (*color, 150)  # Semi-transparent
        
        if self.preview_line.get('is_segment', False):
            # Draw segment insertion preview
            s1 = self.preview_line['s1']
            s2 = self.preview_line['s2']
            target = self.preview_line['target']
            
            # --- FIX STARTS HERE ---
            # Handle both Station objects and dicts for the target's coordinates
            target_x = target.x if hasattr(target, 'x') else target['x']
            target_y = target.y if hasattr(target, 'y') else target['y']

            # Draw dashed lines
            self._draw_dashed_line(preview_surface, s1.x, s1.y, target_x, target_y, alpha_color, CONFIG.LINE_WIDTH)
            self._draw_dashed_line(preview_surface, target_x, target_y, s2.x, s2.y, alpha_color, CONFIG.LINE_WIDTH)
            # --- FIX ENDS HERE ---
        else:
            # Draw line extension preview
            start = self.preview_line['start']
            end = self.preview_line['end']
            
            # Draw dashed line
            self._draw_dashed_line(preview_surface, start['x'], start['y'], end['x'], end['y'], alpha_color, CONFIG.LINE_WIDTH)
        
        screen.blit(preview_surface, (0, 0))

    
    def _draw_dashed_line(self, surface, x1, y1, x2, y2, color, width):
        """Draw a dashed line (matching JS version style)"""
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            return
        
        # Dash pattern: 10px dash, 8px gap (matching JS version)
        dash_length = 10
        gap_length = 8
        total_length = dash_length + gap_length
        
        num_segments = int(distance / total_length)
        
        for i in range(num_segments + 1):
            start_ratio = (i * total_length) / distance
            end_ratio = min((i * total_length + dash_length) / distance, 1.0)
            
            if start_ratio >= 1.0:
                break
            
            start_x = x1 + dx * start_ratio
            start_y = y1 + dy * start_ratio
            end_x = x1 + dx * end_ratio
            end_y = y1 + dy * end_ratio
            
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)