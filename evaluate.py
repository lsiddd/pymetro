# ./evaluate.py
import torch
import time
import pygame
from rl_agent.env import MiniMetroEnv
from rl_agent.dqn_agent import DQNAgent
from rl_agent.rl_config import RLAgentConfig

def render_text(screen, text, position, font, color=(0, 0, 0), bg_color=None):
    """Helper function to render text with an optional background."""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=position)
    if bg_color:
        pygame.draw.rect(screen, bg_color, text_rect.inflate(10, 5))
    screen.blit(text_surface, text_rect.inflate(5, 0)) # Add small padding

def watch_agent(checkpoint_path='checkpoint.pth', episodes=5):
    """Loads a trained agent and runs it in the environment with rendering and on-screen stats."""

    # --- INITIALIZATION ---
    # Initialize the environment with rendering enabled
    env = MiniMetroEnv(headless=False)
    
    # Initialize the agent with a config
    agent_config = RLAgentConfig()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, config=agent_config, seed=0)

    # Load the trained weights
    try:
        agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
        print(f"Successfully loaded model weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Could not find checkpoint file at {checkpoint_path}")
        print("Please train the model first using train.py")
        env.close()
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        env.close()
        return

    # --- EVALUATION LOOP ---
    for i in range(episodes):
        state, valid_actions_mask = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n--- Starting Evaluation Episode {i+1} ---")

        while not done:
            # The agent chooses an action based on its learned policy (epsilon=0 for pure exploitation)
            action = agent.act(state, valid_actions_mask, eps=0.0)
            
            # Perform the action in the environment
            next_state, next_valid_actions_mask, reward_details, done, info = env.step(action)
            
            reward = reward_details['total']
            total_reward += reward
            
            # Update the state
            state = next_state
            valid_actions_mask = next_valid_actions_mask
            
            # --- RENDER GAME AND STATS ---
            env.render()
            
            # Prepare UI elements
            screen = env.screen
            font = pygame.font.Font(None, 28)
            info_y_start = 70
            
            # Display stats on screen
            stats_to_display = {
                f"Episode": f"{i+1}/{episodes}",
                f"Score": f"{info.get('final_score', env.game_state.score)}",
                f"Total Reward": f"{total_reward:.2f}",
                f"Last Reward": f"{reward:.2f}",
                f"Action Result": f"{info.get('action_result', 'N/A')}",
                f"Passengers": f"{info.get('passengers_delivered', env.game_state.passengers_delivered)}",
                f"Days": f"{info.get('days_survived', (env.game_state.week - 1) * 7 + env.game_state.day)}",
            }
            
            for j, (key, value) in enumerate(stats_to_display.items()):
                render_text(screen, f"{key}: {value}", (10, info_y_start + j * 25), font, (0,0,0), (255, 255, 255, 180))

            pygame.display.flip()

            # Add a small delay to make it watchable
            time.sleep(0.1)

    print("Evaluation finished.")
    env.close()

if __name__ == '__main__':
    watch_agent()