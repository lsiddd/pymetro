#!/usr/bin/env python3

import pygame
import sys
import time
from state import game_state
from systems.game import Game
from systems.ui import UI
from systems.input import InputHandler
from systems.ai.ga import GeneticAlgorithmTask

class MiniMetroGame:
    def __init__(self):
        pygame.init()
        
        # Set up display
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Mini Metro - Python Edition")

        # Initialize systems
        self.game = Game()
        self.ui = UI(self.screen)
        self.input_handler = InputHandler()

        # Game loop variables
        self.clock = pygame.time.Clock()
        self.running = True
        self.last_time = time.time()
        
        # AG Thread integration
        self.ga_thread = None
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                self.screen_width = event.w
                self.screen_height = event.h
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                self.ui.update_screen_size(self.screen)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                ui_result = self.ui.handle_click(event.pos)
                if ui_result == 'start_game':
                    self.start_game()
                elif ui_result == 'restart_game':
                    self.restart_game()
                elif ui_result == 'drag_train':
                    self.input_handler.dragged_train_resource = True
                elif ui_result == 'drag_carriage':
                    self.input_handler.dragged_carriage = True
                elif ui_result == 'drag_interchange':
                    self.input_handler.dragged_interchange = True
                elif ui_result == 'start_ga':
                    self.start_ga_optimization()
                elif ui_result == 'start_deep_ga':
                    self.start_deep_ga_optimization()
                elif ui_result == 'instant_spawn':
                    self.game.spawn_station(self.screen_width, self.screen_height)
                elif not ui_result:
                    if self.game.initialized and not self.ui.show_start_screen and not self.ga_thread:
                        world_pos = self._to_world_pos(event.pos)
                        self.input_handler.handle_mouse_down(world_pos, event.button)

            elif event.type == pygame.MOUSEBUTTONUP:
                if self.game.initialized and not self.ui.show_start_screen and not self.ga_thread:
                    world_pos = self._to_world_pos(event.pos)
                    self.input_handler.handle_mouse_up(world_pos, event.button)

            elif event.type == pygame.MOUSEMOTION:
                if self.game.initialized and not self.ui.show_start_screen and not self.ga_thread:
                    world_pos = self._to_world_pos(event.pos)
                    self.input_handler.handle_mouse_motion(world_pos)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game_state.paused = not game_state.paused
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    line_num = event.key - pygame.K_1
                    if line_num < game_state.available_lines:
                        game_state.selected_line = line_num
    
    def _to_world_pos(self, screen_pos):
        """Convert screen position to game-world position accounting for camera zoom."""
        zoom = game_state.camera_zoom
        if zoom >= 0.995:
            return screen_pos
        sw, sh = self.screen_width, self.screen_height
        zw, zh = sw * zoom, sh * zoom
        ox = (sw - zw) / 2
        oy = (sh - zh) / 2
        return ((screen_pos[0] - ox) / zoom, (screen_pos[1] - oy) / zoom)

    def start_game(self):
        """Start a new game"""
        self.game.init_game(self.screen_width, self.screen_height)

    def restart_game(self):
        """Restart the game"""
        if self.ga_thread:
            self.ga_thread.is_running = False
            self.ga_thread = None
        self.ui.show_start_screen = True
        self.ui.show_game_over_modal = False
        game_state.reset()
        
    def start_ga_optimization(self):
        """Start GA algorithm thread"""
        if self.ga_thread is not None and self.ga_thread.is_alive(): return
        game_state.paused = True
        self.ga_thread = GeneticAlgorithmTask(game_state, self._on_ga_complete)
        self.ga_thread.start()
        
    def start_deep_ga_optimization(self):
        """Start deep GA algorithm thread with larger population, more generations and longer horizon"""
        if self.ga_thread is not None and self.ga_thread.is_alive(): return
        game_state.paused = True
        # Set parameters for a deeper search
        self.ga_thread = GeneticAlgorithmTask(
            game_state, 
            self._on_ga_complete,
            pop_size=300,        # Default was 200
            gen_max=100,         # Default was 50
            sim_horizon=10000     # Default was 3000
        )
        self.ga_thread.start()
        
    def _on_ga_complete(self, chromosome):
        """Callback to set up the best chromosome locally. Will be polled in update loop"""
        self._best_chromosome_ready = chromosome
    
    def update(self, delta_time):
        """Update game logic"""
        # Update input handler (mouse pos transformed to world space)
        mouse_pos = self._to_world_pos(pygame.mouse.get_pos())
        mouse_pressed = pygame.mouse.get_pressed()[0]
        self.input_handler.update(mouse_pos, mouse_pressed)
        
        # Apply chromosome if GA just finished
        if getattr(self, '_best_chromosome_ready', None):
            self._apply_ga_chromosome(self._best_chromosome_ready)
            self._best_chromosome_ready = None
            self.ga_thread = None
            game_state.paused = False
        
        # Update game
        if self.game.initialized and not self.ui.show_start_screen and not self.ga_thread:
            result = self.game.update(delta_time, self.screen_width, self.screen_height)
            
            if result == 'show_upgrades':
                self.ui.show_upgrade_choices()
                game_state.paused = True
            elif result == 'game_over':
                self.ui.show_game_over()
                
    def _apply_ga_chromosome(self, chromosome):
        from components.train import Train
        from systems.pathfinding import mark_graph_dirty
        
        # Clear existing lines and trains
        for line in game_state.lines:
            for train in list(line.trains):
                if train in game_state.trains: game_state.trains.remove(train)
                # drop passengers back to starting stations
                for passg in train.passengers:
                    if passg in game_state.passengers: game_state.passengers.remove(passg)
            line.trains.clear()
            line.stations.clear()
            line.active = False
            line.marked_for_deletion = False

        # Apply new ones
        station_map = {s.id: s for s in game_state.stations}
        for i, line_stations in enumerate(chromosome.lines):
            line = game_state.lines[i]
            # Must populate actual station instances since chromosome may have only id's in the future
            if line_stations:
                line.stations = [station_map[sid] for sid in line_stations]
                line.active = len(line.stations) >= 2
                
            game_state.available_trains = 0 
            game_state.carriages = 0 
        
        num_trains_total = sum(chromosome.trains_per_line)
        num_car_total = sum(chromosome.carriages_per_line)
        game_state.available_trains += max(0, 3 - num_trains_total) # dummy logic for leftover
        
        for i, line in enumerate(game_state.lines):
            if not line.active: continue
            train_count = chromosome.trains_per_line[i]
            car_count = chromosome.carriages_per_line[i]
            
            for t_idx in range(train_count):
                train_carriages = car_count // train_count
                if t_idx < car_count % train_count: train_carriages += 1
                
                t = Train(line)
                t.carriage_count = train_carriages
                game_state.trains.append(t)
                line.trains.append(t)
                
        mark_graph_dirty()
    
    def render(self):
        """Render the game"""
        if self.game.initialized and not self.ui.show_start_screen:
            world_surf = self.game.render(self.screen)

            if world_surf is not None:
                # Draw input preview onto the world surface so it scales with
                # the rest of the world (fixes misalignment at zoom < 1).
                self.input_handler.draw_preview(world_surf)

                zoom = game_state.camera_zoom
                sw, sh = self.screen_width, self.screen_height
                if zoom < 0.995:
                    self.screen.fill((210, 212, 208))
                    zw, zh = int(sw * zoom), int(sh * zoom)
                    scaled = pygame.transform.smoothscale(world_surf, (zw, zh))
                    self.screen.blit(scaled, ((sw - zw) // 2, (sh - zh) // 2))
                else:
                    self.screen.blit(world_surf, (0, 0))

        # Render UI on top
        self.ui.draw()
        
        # Render GA overlay
        if self.ga_thread and self.ga_thread.is_alive():
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            font = pygame.font.Font(None, 48)
            txt = font.render(f"Configurando Topologia... {int(self.ga_thread.progress * 100)}%", True, (255,255,255))
            trank = txt.get_rect(center=(self.screen_width//2, self.screen_height//2))
            self.screen.blit(txt, trank)

        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = (current_time - self.last_time) * 1000  # Convert to milliseconds
            self.last_time = current_time
            
            # Cap delta time to prevent large jumps
            if delta_time > 50:
                delta_time = 50
            
            # Handle events
            self.handle_events()
            
            # Update
            self.update(delta_time)
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

def main():
    """Main entry point"""
    try:
        game = MiniMetroGame()
        game.run()
    except Exception:
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()