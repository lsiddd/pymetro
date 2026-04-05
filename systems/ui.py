# ./systems/ui.py
import pygame
import math
import time
import random
from config import CONFIG

class UI:
    def __init__(self, screen):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        # self.font = pygame.font.Font(None, 24)
        # self.large_font = pygame.font.Font(None, 48)
        # self.small_font = pygame.font.Font(None, 18)

        try:
            # On Windows, 'Segoe UI Emoji' is often available
            self.font = pygame.font.Font("seguiemj.ttf", 24)
            self.large_font = pygame.font.Font("seguiemj.ttf", 48)
            self.small_font = pygame.font.Font("seguiemj.ttf", 18)
        except FileNotFoundError:
            # Fallback for other systems or if the font isn't installed.
            # You should provide a font file like NotoEmoji-Regular.ttf with your game.
            try:
                self.font = pygame.font.Font("NotoEmoji-Regular.ttf", 24)
                self.large_font = pygame.font.Font("NotoEmoji-Regular.ttf", 48)
                self.small_font = pygame.font.Font("NotoEmoji-Regular.ttf", 18)
            except FileNotFoundError:
                 print("Emoji font not found! Please add a .ttf font file to the project.")
                 self.font = pygame.font.Font(None, 24)
                 self.large_font = pygame.font.Font(None, 48)
                 self.small_font = pygame.font.Font(None, 18)
        
        # UI state
        self.show_upgrade_modal = False
        self.upgrade_choices = []
        self.show_game_over_modal = False
        self.show_start_screen = True
        
        # Button rects
        self.pause_btn_rect = pygame.Rect(self.width - 100, 10, 40, 40)
        self.speed_btn_rect = pygame.Rect(self.width - 50, 10, 40, 40)
        self.line_selector_rects = []
        self.line_delete_rects = []
        self.city_btn_rects = {}
        self.start_btn_rect = None
        self.restart_btn_rect = None

        # Draggable resource icon rects (updated each frame in draw_resources)
        self.train_resource_rect = None
        self.carriage_resource_rect = None
        self.interchange_resource_rect = None
        
    def update_screen_size(self, screen):
        """Update UI for new screen size"""
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        
        # Update button positions
        self.pause_btn_rect = pygame.Rect(self.width - 100, 10, 40, 40)
        self.speed_btn_rect = pygame.Rect(self.width - 50, 10, 40, 40)
    
    def draw(self):
        """Draw all UI elements"""
        from state import game_state
        
        if self.show_start_screen:
            self.draw_start_screen()
        elif self.show_game_over_modal:
            self.draw_game_over_modal()
        elif self.show_upgrade_modal:
            self.draw_upgrade_modal()
        else:
            self.draw_game_ui()
    
    def draw_start_screen(self):
        """Draw the start screen"""
        from state import game_state
        
        # Background
        self.screen.fill((244, 241, 233))
        
        # Game title
        title_text = self.large_font.render("MINI METRO", True, (51, 51, 51))
        title_rect = title_text.get_rect(center=(self.width//2, self.height//2 - 150))
        self.screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = self.font.render("Design the subway system", True, (102, 102, 102))
        subtitle_rect = subtitle_text.get_rect(center=(self.width//2, self.height//2 - 100))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # City selection
        city_label = self.font.render("Select City", True, (51, 51, 51))
        city_label_rect = city_label.get_rect(center=(self.width//2, self.height//2 - 50))
        self.screen.blit(city_label, city_label_rect)
        
        # City buttons
        cities = ['london', 'paris', 'newyork', 'tokyo']
        city_names = ['London', 'Paris', 'New York', 'Tokyo']
        
        self.city_btn_rects = {}
        for i, (city, name) in enumerate(zip(cities, city_names)):
            x = self.width//2 - 150 + (i % 2) * 150
            y = self.height//2 - 10 + (i // 2) * 50
            
            rect = pygame.Rect(x, y, 120, 40)
            self.city_btn_rects[city] = rect
            
            color = (51, 51, 51) if game_state.selected_city == city else (255, 255, 255)
            text_color = (255, 255, 255) if game_state.selected_city == city else (51, 51, 51)
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (51, 51, 51), rect, 2)
            
            city_text = self.font.render(name, True, text_color)
            city_text_rect = city_text.get_rect(center=rect.center)
            self.screen.blit(city_text, city_text_rect)
        
        # Start button
        self.start_btn_rect = pygame.Rect(self.width//2 - 75, self.height//2 + 100, 150, 50)
        pygame.draw.rect(self.screen, (76, 175, 80), self.start_btn_rect)
        pygame.draw.rect(self.screen, (51, 51, 51), self.start_btn_rect, 2)
        
        start_text = self.font.render("START", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=self.start_btn_rect.center)
        self.screen.blit(start_text, start_text_rect)
    
    def draw_game_ui(self):
        """Draw the main game UI"""
        from state import game_state
        
        # Top bar background
        top_bar_rect = pygame.Rect(0, 0, self.width, 60)
        top_bar_surface = pygame.Surface((self.width, 60), pygame.SRCALPHA)
        pygame.draw.rect(top_bar_surface, (244, 241, 233, 230), top_bar_rect)
        self.screen.blit(top_bar_surface, (0, 0))
        
        # Clock
        self.draw_clock()
        
        # Score
        score_text = self.font.render(str(game_state.score), True, (51, 51, 51))
        score_bg_rect = pygame.Rect(self.width//2 - 40, 70, 80, 35)
        pygame.draw.rect(self.screen, (255, 255, 255, 200), score_bg_rect)
        pygame.draw.rect(self.screen, (51, 51, 51), score_bg_rect, 2)
        
        score_rect = score_text.get_rect(center=score_bg_rect.center)
        self.screen.blit(score_text, score_rect)
        
        # Resources
        self.draw_resources()
        
        # Control buttons
        self.draw_control_buttons()
        
        # Line selector
        self.draw_line_selector()
    
    def draw_clock(self):
        """Draw the week/day clock"""
        from state import game_state
        
        # Clock circle
        clock_center = (50, 30)
        pygame.draw.circle(self.screen, (221, 221, 221), clock_center, 25, 3)
        
        # Progress arc
        elapsed = time.time() * 1000 - game_state.week_start_time
        progress = (elapsed % CONFIG.WEEK_DURATION) / CONFIG.WEEK_DURATION
        
        if progress > 0:
            angle = progress * 2 * math.pi - math.pi/2
            pygame.draw.arc(self.screen, (51, 51, 51), 
                           (25, 5, 50, 50), -math.pi/2, angle, 3)
        
        # Clock text
        days = ['M', 'Tu', 'W', 'Th', 'F', 'Sa', 'Su']
        week_text = f"W{game_state.week}"
        day_text = days[game_state.day] if game_state.day < len(days) else 'M'
        
        week_surface = self.small_font.render(week_text, True, (51, 51, 51))
        day_surface = self.small_font.render(day_text, True, (51, 51, 51))
        
        week_rect = week_surface.get_rect(center=(clock_center[0], clock_center[1] - 5))
        day_rect = day_surface.get_rect(center=(clock_center[0], clock_center[1] + 8))
        
        self.screen.blit(week_surface, week_rect)
        self.screen.blit(day_surface, day_rect)
    
    def draw_resources(self):
        """Draw resource counters. Locomotives, carriages and interchanges are draggable."""
        from state import game_state

        # Reset draggable rects each frame
        self.train_resource_rect = None
        self.carriage_resource_rect = None
        self.interchange_resource_rect = None

        resources = [
            ('train',       '🚂', game_state.available_trains, True),
            ('carriage',    '🚃', game_state.carriages,        game_state.carriages > 0),
            ('interchange', '⭕', game_state.interchanges,     game_state.interchanges > 0),
            ('bridge',      '🌉', game_state.bridges,          game_state.bridges > 0),
        ]

        x = self.width // 2 - 100
        for key, icon, count, visible in resources:
            if not visible:
                continue

            resource_rect = pygame.Rect(x, 15, 60, 30)

            # Highlight draggable resources
            is_draggable = key in ('train', 'carriage', 'interchange') and count > 0
            bg_color = (220, 240, 255) if is_draggable else (255, 255, 255)
            border_color = (50, 120, 200) if is_draggable else (51, 51, 51)

            pygame.draw.rect(self.screen, bg_color, resource_rect)
            pygame.draw.rect(self.screen, border_color, resource_rect, 2)

            icon_text  = self.font.render(icon, True, (51, 51, 51))
            count_text = self.small_font.render(str(count), True, (51, 51, 51))
            self.screen.blit(icon_text,  (x + 5,  20))
            self.screen.blit(count_text, (x + 35, 25))

            # Store rects so input handler can detect drags
            if key == 'train':
                self.train_resource_rect = resource_rect
            elif key == 'carriage':
                self.carriage_resource_rect = resource_rect
            elif key == 'interchange':
                self.interchange_resource_rect = resource_rect

            x += 70
    
    def draw_control_buttons(self):
        """Draw pause and speed control buttons"""
        from state import game_state
        
        # Pause button
        pause_color = (51, 51, 51) if game_state.paused else (255, 255, 255)
        text_color = (255, 255, 255) if game_state.paused else (51, 51, 51)
        
        pygame.draw.circle(self.screen, pause_color, self.pause_btn_rect.center, 20)
        pygame.draw.circle(self.screen, (51, 51, 51), self.pause_btn_rect.center, 20, 2)
        
        pause_text = "▶" if game_state.paused else "⏸"
        pause_surface = self.font.render(pause_text, True, text_color)
        pause_text_rect = pause_surface.get_rect(center=self.pause_btn_rect.center)
        self.screen.blit(pause_surface, pause_text_rect)
        
        # Speed button
        speed_color = (51, 51, 51) if game_state.speed != 1 else (255, 255, 255)
        text_color = (255, 255, 255) if game_state.speed != 1 else (51, 51, 51)
        
        pygame.draw.circle(self.screen, speed_color, self.speed_btn_rect.center, 20)
        pygame.draw.circle(self.screen, (51, 51, 51), self.speed_btn_rect.center, 20, 2)
        
        speed_text = "⏩" if game_state.speed != 1 else "▶"
        speed_surface = self.font.render(speed_text, True, text_color)
        speed_text_rect = speed_surface.get_rect(center=self.speed_btn_rect.center)
        self.screen.blit(speed_surface, speed_text_rect)
    
    def draw_line_selector(self):
        """Draw line selection buttons at bottom"""
        from state import game_state
        
        # Background
        selector_width = min(game_state.available_lines * 60, 400)
        selector_rect = pygame.Rect(self.width//2 - selector_width//2, self.height - 80, 
                                  selector_width, 60)
        
        pygame.draw.rect(self.screen, (255, 255, 255, 200), selector_rect)
        pygame.draw.rect(self.screen, (51, 51, 51), selector_rect, 2)
        
        # Line buttons
        self.line_selector_rects = []
        # --- CHANGE START ---
        self.line_delete_rects = [] # Reset the list each frame
        # --- CHANGE END ---
        for i in range(game_state.available_lines):
            x = self.width//2 - (game_state.available_lines * 30) + i * 60 + 30
            y = self.height - 50
            
            button_rect = pygame.Rect(x - 22, y - 22, 44, 44)
            self.line_selector_rects.append(button_rect)
            
            # Button appearance
            if i == game_state.selected_line:
                pygame.draw.circle(self.screen, game_state.lines[i].color, (x, y), 25)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 25, 3)
            else:
                pygame.draw.circle(self.screen, game_state.lines[i].color, (x, y), 20)
                pygame.draw.circle(self.screen, (51, 51, 51), (x, y), 20, 2)
            
            # Delete button for active lines
            line = game_state.lines[i]
            if line.active and len(line.stations) > 0:
                delete_center_x = x + 16
                delete_center_y = y - 12
                delete_rect = pygame.Rect(delete_center_x - 8, delete_center_y - 8, 16, 16)
                # --- CHANGE START ---
                self.line_delete_rects.append((delete_rect, i)) # Store rect and line index
                # --- CHANGE END ---
                
                pygame.draw.circle(self.screen, (231, 76, 60), (delete_center_x, delete_center_y), 8)
                pygame.draw.circle(self.screen, (255, 255, 255), (delete_center_x, delete_center_y), 8, 2)
                
                delete_text = self.small_font.render("×", True, (255, 255, 255))
                delete_text_rect = delete_text.get_rect(center=(delete_center_x, delete_center_y))
                self.screen.blit(delete_text, delete_text_rect)
    
    def draw_upgrade_modal(self):
        """Draw upgrade selection modal.
        Shows the guaranteed locomotive reward and two clickable options."""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        modal_rect = pygame.Rect(self.width // 2 - 280, self.height // 2 - 170, 560, 340)
        pygame.draw.rect(self.screen, (255, 255, 255), modal_rect)
        pygame.draw.rect(self.screen, (51, 51, 51), modal_rect, 3)

        # Title
        title = self.large_font.render("Weekly Rewards", True, (51, 51, 51))
        self.screen.blit(title, title.get_rect(center=(self.width // 2, self.height // 2 - 130)))

        # Guaranteed locomotive (left column, greyed-out / already applied)
        loco_rect = pygame.Rect(self.width // 2 - 260, self.height // 2 - 90, 130, 120)
        pygame.draw.rect(self.screen, (230, 245, 230), loco_rect)
        pygame.draw.rect(self.screen, (100, 180, 100), loco_rect, 3)

        loco_icon = self.large_font.render('🚂', True, (51, 51, 51))
        self.screen.blit(loco_icon, loco_icon.get_rect(center=(loco_rect.centerx, loco_rect.centery - 18)))
        loco_lbl = self.small_font.render('Locomotive', True, (51, 51, 51))
        self.screen.blit(loco_lbl, loco_lbl.get_rect(center=(loco_rect.centerx, loco_rect.centery + 28)))
        received = self.small_font.render('(received)', True, (100, 160, 100))
        self.screen.blit(received, received.get_rect(center=(loco_rect.centerx, loco_rect.centery + 44)))

        # Divider + label
        div_x = self.width // 2 - 115
        pygame.draw.line(self.screen, (180, 180, 180), (div_x, self.height // 2 - 85), (div_x, self.height // 2 + 35), 2)
        choose_lbl = self.font.render("Choose one:", True, (80, 80, 80))
        self.screen.blit(choose_lbl, choose_lbl.get_rect(center=(self.width // 2 + 95, self.height // 2 - 105)))

        # Two clickable choices (right side)
        choice_w, choice_h = 130, 120
        self.upgrade_rects = []
        for i, upgrade in enumerate(self.upgrade_choices):
            x = self.width // 2 - 100 + i * (choice_w + 20)
            y = self.height // 2 - 90
            rect = pygame.Rect(x, y, choice_w, choice_h)
            self.upgrade_rects.append((rect, upgrade))

            pygame.draw.rect(self.screen, (245, 245, 255), rect)
            pygame.draw.rect(self.screen, (51, 51, 51), rect, 3)

            content = self.get_upgrade_content(upgrade)
            icon = self.large_font.render(content['icon'], True, (51, 51, 51))
            self.screen.blit(icon, icon.get_rect(center=(rect.centerx, rect.centery - 18)))
            name = self.small_font.render(content['name'], True, (51, 51, 51))
            self.screen.blit(name, name.get_rect(center=(rect.centerx, rect.centery + 30)))
    
    def draw_game_over_modal(self):
        """Draw game over modal"""
        from state import game_state
        
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Modal background
        modal_rect = pygame.Rect(self.width//2 - 250, self.height//2 - 150, 500, 300)
        pygame.draw.rect(self.screen, (255, 255, 255), modal_rect)
        pygame.draw.rect(self.screen, (51, 51, 51), modal_rect, 3)
        
        # Title
        title_text = self.large_font.render("Metro Closed", True, (211, 47, 47))
        title_rect = title_text.get_rect(center=(self.width//2, self.height//2 - 100))
        self.screen.blit(title_text, title_rect)
        
        # Stats
        game_time = time.time() * 1000 - game_state.game_start_time
        days = int(game_time / CONFIG.WEEK_DURATION * 7)
        
        stats = [
            f"Passengers Transported: {game_state.passengers_delivered}",
            f"Days Survived: {days}",
            f"Final Score: {game_state.score}"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = self.font.render(stat, True, (51, 51, 51))
            stat_rect = stat_text.get_rect(center=(self.width//2, self.height//2 - 40 + i * 30))
            self.screen.blit(stat_text, stat_rect)
        
        # Restart button
        self.restart_btn_rect = pygame.Rect(self.width//2 - 75, self.height//2 + 60, 150, 40)
        pygame.draw.rect(self.screen, (51, 51, 51), self.restart_btn_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.restart_btn_rect, 2)
        
        restart_text = self.font.render("Play Again", True, (255, 255, 255))
        restart_text_rect = restart_text.get_rect(center=self.restart_btn_rect.center)
        self.screen.blit(restart_text, restart_text_rect)
    
    def get_upgrade_content(self, upgrade):
        """Get display content for upgrade type"""
        content = {
            'line': {'icon': '🚇', 'name': 'New Line'},
            'tunnel': {'icon': '🌉', 'name': 'Bridge/Tunnel'},
            'carriage': {'icon': '🚃', 'name': 'Carriage'},
            'interchange': {'icon': '🔄', 'name': 'Interchange'}
        }
        return content.get(upgrade, {'icon': '?', 'name': 'Unknown'})
    
    def show_upgrade_choices(self):
        """Show upgrade modal. Locomotive is always granted; player picks 1 of 2 random extras."""
        from state import game_state

        pool = ['line', 'tunnel', 'carriage', 'interchange']
        available = [u for u in pool if u != 'line' or game_state.available_lines < game_state.max_lines]
        self.upgrade_choices = random.sample(available, min(2, len(available)))
        self.show_upgrade_modal = True
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        from state import game_state
        
        if self.show_start_screen:
            return self.handle_start_screen_click(pos)
        elif self.show_game_over_modal:
            return self.handle_game_over_click(pos)
        elif self.show_upgrade_modal:
            return self.handle_upgrade_click(pos)
        else:
            return self.handle_game_ui_click(pos)
    
    def handle_start_screen_click(self, pos):
        """Handle clicks on start screen"""
        from state import game_state
        
        # City selection
        for city, rect in self.city_btn_rects.items():
            if rect.collidepoint(pos):
                game_state.set_city(city)
                return True
        
        # Start button
        if self.start_btn_rect and self.start_btn_rect.collidepoint(pos):
            self.show_start_screen = False
            return 'start_game'
        
        return False
    
    def handle_game_over_click(self, pos):
        """Handle clicks on game over modal"""
        if self.restart_btn_rect and self.restart_btn_rect.collidepoint(pos):
            self.show_game_over_modal = False
            self.show_start_screen = True
            return 'restart_game'
        return False
    
    def handle_upgrade_click(self, pos):
        """Handle clicks on upgrade modal"""
        from state import game_state
        
        for rect, upgrade in getattr(self, 'upgrade_rects', []):
            if rect.collidepoint(pos):
                self.apply_upgrade(upgrade)
                self.show_upgrade_modal = False
                game_state.paused = False
                return True
        return False
    
    def handle_game_ui_click(self, pos):
        """Handle clicks on game UI"""
        from state import game_state

        # Control buttons
        if self.pause_btn_rect.collidepoint(pos):
            game_state.paused = not game_state.paused
            return True

        if self.speed_btn_rect.collidepoint(pos):
            game_state.speed = 2.5 if game_state.speed == 1 else 1
            return True

        # Line delete buttons
        for rect, line_index in self.line_delete_rects:
            if rect.collidepoint(pos):
                line = game_state.lines[line_index]
                if line.active:
                    line.marked_for_deletion = True
                    return True

        # Line selector
        for i, rect in enumerate(self.line_selector_rects):
            if rect.collidepoint(pos):
                game_state.selected_line = i
                return True

        # Draggable resource icons — return tag so main.py can start a drag
        if self.train_resource_rect and self.train_resource_rect.collidepoint(pos):
            if game_state.available_trains > 0:
                return 'drag_train'
        if self.carriage_resource_rect and self.carriage_resource_rect.collidepoint(pos):
            if game_state.carriages > 0:
                return 'drag_carriage'
        if self.interchange_resource_rect and self.interchange_resource_rect.collidepoint(pos):
            if game_state.interchanges > 0:
                return 'drag_interchange'

        return False
    
    def apply_upgrade(self, upgrade):
        """Apply the chosen optional upgrade."""
        from state import game_state

        if upgrade == 'line':
            game_state.available_lines += 1
        elif upgrade == 'tunnel':
            game_state.bridges += 1
        elif upgrade == 'carriage':
            game_state.carriages += 1
        elif upgrade == 'interchange':
            game_state.interchanges += 1
    
    def show_game_over(self):
        """Show game over modal"""
        self.show_game_over_modal = True

# Global function for resource display updates (matching JS version)
def update_resource_display():
    """Update resource display - placeholder for now"""
    from state import game_state
    print(f"Resources: Trains={game_state.available_trains}, Bridges={game_state.bridges}, "
          f"Carriages={game_state.carriages}, Interchanges={game_state.interchanges}")