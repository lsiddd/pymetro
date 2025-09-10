#!/usr/bin/env python3

import pygame
import sys
import time
from config import CONFIG
from state import game_state
from systems.game import Game
from systems.ui import UI
from systems.input import InputHandler

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
        
        print("Mini Metro - Python Edition initialized!")
        print("Controls:")
        print("- Left click: Add stations to lines, select UI elements")
        print("- Right click: Remove stations from lines")
        print("- Mouse hover: Preview line connections")
        print("- Drag resources to lines/stations (trains, carriages, interchanges)")
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                print("MiniMetro: Quit event received")
            
            elif event.type == pygame.VIDEORESIZE:
                self.screen_width = event.w
                self.screen_height = event.h
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
                self.ui.update_screen_size(self.screen)
                print(f"MiniMetro: Window resized to {self.screen_width}x{self.screen_height}")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(f"MiniMetro: Mouse button {event.button} down at {event.pos}")
                # First check UI clicks
                ui_result = self.ui.handle_click(event.pos)
                if ui_result == 'start_game':
                    self.start_game()
                elif ui_result == 'restart_game':
                    self.restart_game()
                elif ui_result:
                    print(f"MiniMetro: UI handled click: {ui_result}")
                else:
                    # Handle game world clicks
                    if self.game.initialized and not self.ui.show_start_screen:
                        self.input_handler.handle_mouse_down(event.pos, event.button)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                print(f"MiniMetro: Mouse button {event.button} up at {event.pos}")
                if self.game.initialized and not self.ui.show_start_screen:
                    self.input_handler.handle_mouse_up(event.pos, event.button)
            
            elif event.type == pygame.MOUSEMOTION:
                if self.game.initialized and not self.ui.show_start_screen:
                    self.input_handler.handle_mouse_motion(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                print(f"MiniMetro: Key pressed: {pygame.key.name(event.key)}")
                if event.key == pygame.K_SPACE:
                    game_state.paused = not game_state.paused
                    print(f"MiniMetro: Game {'paused' if game_state.paused else 'resumed'}")
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    # Quick line selection
                    line_num = event.key - pygame.K_1
                    if line_num < game_state.available_lines:
                        game_state.selected_line = line_num
                        print(f"MiniMetro: Selected line {line_num}")
    
    def start_game(self):
        """Start a new game"""
        self.game.init_game(self.screen_width, self.screen_height)
        print(f"Started new game in {game_state.selected_city.title()}")
    
    def restart_game(self):
        """Restart the game"""
        self.ui.show_start_screen = True
        self.ui.show_game_over_modal = False
        game_state.reset()
        print("Game restarted")
    
    def update(self, delta_time):
        """Update game logic"""
        # Update input handler
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        self.input_handler.update(mouse_pos, mouse_pressed)
        
        # Update game
        if self.game.initialized and not self.ui.show_start_screen:
            result = self.game.update(delta_time, self.screen_width, self.screen_height)
            
            if result == 'show_upgrades':
                self.ui.show_upgrade_choices()
                game_state.paused = True
            elif result == 'game_over':
                self.ui.show_game_over()
    
    def render(self):
        """Render the game"""
        # Render game world
        if self.game.initialized and not self.ui.show_start_screen:
            self.game.render(self.screen)
            
            # Draw input preview
            self.input_handler.draw_preview(self.screen)
        
        # Render UI on top
        self.ui.draw()
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("Starting game loop...")
        
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
        
        print("Game shutting down...")
        pygame.quit()
        sys.exit()

def main():
    """Main entry point"""
    try:
        game = MiniMetroGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()