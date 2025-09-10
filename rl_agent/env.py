# ./rl_agent/env.py
import pygame
import numpy as np

from systems.game import Game
from state import game_state
from rl_agent.state import get_state_vector, STATE_SIZE
from rl_agent.action import perform_action, ACTION_SIZE, get_valid_actions
from rl_agent.reward import calculate_reward, update_previous_metrics

class MiniMetroEnv:
    def __init__(self, headless=True):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.headless = headless
        
        if not self.headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Mini Metro RL Environment")
        else:
            self.screen = None # No screen needed in headless mode
        
        self.game = Game()
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self._max_episode_steps = 5000
        self._current_step = 0
        self.game_state = game_state

    def reset(self):
        self.game.init_game(self.screen_width, self.screen_height)
        self._current_step = 0
        update_previous_metrics() # Initialize metrics for reward calculation
        state_vector = get_state_vector()
        valid_actions_mask = get_valid_actions()
        return state_vector, valid_actions_mask

    def step(self, action):
        self._current_step += 1
        
        # 1. Perform Action
        action_result = perform_action(action)

        # 2. Update Game State
        # Simulate a few frames for the action's effect to be measurable
        num_frames_to_simulate = 30 
        delta_time = 16.67 # Corresponds to 60 FPS
        
        for _ in range(num_frames_to_simulate):
            if self.game_state.game_over:
                break
            game_status = self.game.update(delta_time, self.screen_width, self.screen_height)
            if game_status == 'game_over':
                break
        
        if not self.headless and self._current_step % 5 == 0:
            self.render()

        # 3. Get Next State and Valid Actions
        next_state = get_state_vector()
        next_valid_actions_mask = get_valid_actions()

        # 4. Calculate Reward
        reward_details = calculate_reward(action_result)

        # 5. Check if Done
        done = False
        info = {'action_result': action_result} # Initialize info dict
        if self.game_state.game_over or self._current_step >= self._max_episode_steps:
            done = True
            if self.game_state.game_over:
                 # Add heavy penalty for losing to the reward dictionary
                 reward_details['total'] -= 500.0
                 reward_details['game_over_penalty'] = -500.0

            # --- ADD FINAL METRICS TO INFO DICT ---
            info['passengers_delivered'] = self.game_state.passengers_delivered
            info['final_score'] = self.game_state.score
            info['days_survived'] = (self.game_state.week - 1) * 7 + self.game_state.day
        
        info['valid_actions_count'] = np.sum(next_valid_actions_mask)

        # 6. Update metrics for next step's reward calculation
        update_previous_metrics()

        return next_state, next_valid_actions_mask, reward_details, done, info

    def render(self):
        if self.headless or self.screen is None:
            return
        
        # This allows closing the window during evaluation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit() # Exit the script if window is closed

        self.game.render(self.screen)
        # Flip is handled by the evaluate script to allow drawing stats over the game
        # pygame.display.flip() 

    def close(self):
        pygame.quit()