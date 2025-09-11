# ./train.py

from typing import Optional
import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import os
import multiprocessing as mp

# --- CHANGE START ---
from config import CONFIG # Import CONFIG to get access to difficulty levels
# --- CHANGE END ---

# Set multiprocessing start method for compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

from rl_agent.parallel_env import ParallelEnvironments
from rl_agent.dqn_agent import DQNAgent
from rl_agent.rl_config import RLAgentConfig, TrainingConfig, ParallelRLAgentConfig
# This import is removed from here: from rl_agent.env import MiniMetroEnv

# --- CHANGE START ---
def evaluate_policy(agent: 'DQNAgent', num_eval_episodes: int = 5, difficulty_stage: int = 0) -> float:
    """Evaluates the agent's policy for a number of episodes at a specific difficulty."""
    # The import is moved inside the function that uses it
    from rl_agent.env import MiniMetroEnv
    eval_env = MiniMetroEnv(headless=True)
    total_score = 0
    
    for _ in range(num_eval_episodes):
        state, valid_actions_mask = eval_env.reset(difficulty_stage=difficulty_stage)
        done = False
        episode_score = 0
        
        while not done:
            action = agent.act(state, valid_actions_mask, eps=0.0)  # Greedy policy
            next_state, next_valid_actions_mask, reward_details, done, info = eval_env.step(action)
            episode_score += reward_details['total']
            state = next_state
            valid_actions_mask = next_valid_actions_mask
        
        total_score += info.get('final_score', 0)
    
    eval_env.close()
    return total_score / num_eval_episodes
# --- CHANGE END ---


def train_parallel(config: 'TrainingConfig', agent_config: 'RLAgentConfig', num_workers: int = 4) -> None:
    """Parallel Deep Q-Learning training with multiple environments."""
    
    # --- INITIALIZATION ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", f"minimetro_parallel_dqn_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Training with {num_workers} parallel environments")
    
    # Save configuration
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        f.write(f"--- Training Config ---\n{str(config)}\n")
        f.write(f"--- Agent Config ---\n{str(agent_config)}\n")
        f.write(f"--- Parallel Config ---\n")
        f.write(f"Number of Workers: {num_workers}\n")

    # Initialize parallel environments
    print("Initializing parallel environments...")
    parallel_envs: 'ParallelEnvironments' = ParallelEnvironments(num_workers=num_workers)
    print("Parallel environments initialized successfully")
    
    # Initialize agent
    print("Initializing DQN agent...")
    agent: 'DQNAgent' = DQNAgent(
        state_size=parallel_envs.state_size, 
        action_size=parallel_envs.action_size, 
        config=agent_config, 
        seed=0
    )
    print("DQN agent initialized successfully")

    # Training metrics
    scores_window: deque = deque(maxlen=100)
    last_save_time: float = time.time()
    
    # --- CHANGE START ---
    # Evaluation and Checkpointing
    best_score: float = -float('inf')
    best_episode: int = 0
    evaluation_interval: int = 100  # Evaluate every 100 episodes
    
    # Curriculum Learning
    current_stage: int = 0 
    max_stage: int = len(CONFIG.DIFFICULTY_LEVELS) - 1
    stage_thresholds: list = [20, 50, 100, 200] # Score thresholds to advance to the next stage
    parallel_envs.set_difficulty(current_stage)
    print(f"Starting at curriculum stage {current_stage}")
    # --- CHANGE END ---
    
    # Episode tracking
    episode_count: int = 0
    total_steps: int = 0
    
    # Per-worker episode tracking
    worker_episode_rewards: list = [0.0] * num_workers
    worker_episode_steps: list = [0] * num_workers
    worker_episode_passengers: list = [0] * num_workers
    worker_episode_valid_actions: list = [[] for _ in range(num_workers)]
    
    print("--- Starting Parallel Training ---")
    print(f"Agent will be saved every {config.SAVE_INTERVAL_EPISODES} episodes or {config.SAVE_INTERVAL_SECONDS / 60:.0f} minutes.")

    # Initialize all environments
    print("Resetting all environments...")
    states, valid_actions_masks = parallel_envs.reset_all()
    print("All environments reset successfully")
    
    # --- TRAINING LOOP ---
    print("Entering training loop...")
    loop_step_counter: int = 0
    while episode_count < config.N_EPISODES:
        loop_step_counter += 1
        # Get actions for all environments (using noisy networks)
        actions = agent.act_batch(states, valid_actions_masks)
        
        # Step all environments
        next_states, next_valid_actions_masks, experiences, reward_details_list, dones, infos = parallel_envs.step_all(actions)
        
        # Add experiences to replay buffer and potentially learn
        agent.step_batch(experiences)
        
        # Update states
        states = next_states
        valid_actions_masks = next_valid_actions_masks
        
        # Update per-worker metrics
        for i in range(num_workers):
            worker_episode_rewards[i] += reward_details_list[i]['total']
            worker_episode_steps[i] += 1
            worker_episode_valid_actions[i].append(infos[i].get('valid_actions_count', 0))
            
            # Check if this environment finished an episode
            if dones[i]:
                episode_count += 1
                total_steps += worker_episode_steps[i]
                
                # Extract final metrics
                final_passengers = infos[i].get('passengers_delivered', 0)
                final_score = infos[i].get('final_score', 0)
                days_survived = infos[i].get('days_survived', 0)
                
                worker_episode_passengers[i] = final_passengers
                
                # Add to scores window
                scores_window.append(final_score)
                
                # --- TENSORBOARD LOGGING (ENHANCED) ---
                writer.add_scalar('Performance/Episode Reward', worker_episode_rewards[i], episode_count)
                writer.add_scalar('Performance/Average Score (100 episodes)', np.mean(scores_window), episode_count)
                writer.add_scalar('Game/Passengers Delivered', final_passengers, episode_count)
                writer.add_scalar('Game/Final Score', final_score, episode_count)
                writer.add_scalar('Game/Days Survived', days_survived, episode_count)
                writer.add_scalar('Agent/Replay_Buffer_Size', len(agent.memory), episode_count)
                writer.add_scalar('Agent/Episode Length (steps)', worker_episode_steps[i], episode_count)
                
                # --- CHANGE START ---
                writer.add_scalar('Curriculum/Difficulty_Stage', current_stage, episode_count)
                # --- CHANGE END ---
                
                # New enhanced logging
                writer.add_scalar('Agent/Learning_Rate', agent.optimizer.param_groups[0]['lr'], episode_count)
                writer.add_scalar('Agent/Gradient_Norm', agent.last_gradient_norm, episode_count)
                avg_valid_actions = np.mean(worker_episode_valid_actions[i]) if worker_episode_valid_actions[i] else 0
                writer.add_scalar('Environment/Average_Valid_Actions', avg_valid_actions, episode_count)
                writer.add_scalar('Environment/Stations_Count', infos[i].get('stations_count', 0), episode_count)
                writer.add_scalar('Environment/Connected_Ratio', infos[i].get('connectivity_ratio', 0.0), episode_count)

                if agent.last_loss is not None:
                    writer.add_scalar('Agent/Loss', agent.last_loss, episode_count)
                
                # Histogram logging for better debugging
                if episode_count % 100 == 0:
                    if agent.last_q_values is not None:
                        writer.add_histogram('Agent/Q_Values', agent.last_q_values, episode_count)
                    writer.add_histogram('Agent/Layer1_Weights', agent.qnetwork_local.shared_fc1.weight, episode_count)
                    writer.add_histogram('Agent/Layer2_Weights', agent.qnetwork_local.shared_fc2.weight, episode_count)

                # --- CONSOLE LOGGING ---
                print_str = (
                    f"Episode {episode_count:4d}\t"
                    f"Avg Score: {np.mean(scores_window):7.2f}\t"
                    f"Passengers: {final_passengers:3d}\t"
                    f"Days: {days_survived:3d}\t"
                    f"Buffer: {len(agent.memory):6d}\t"
                    f"Stage: {current_stage}\t" # --- CHANGE ---
                    f"Best Ep: {best_episode} ({best_score:.2f})"
                )
                print(f'\r{print_str}', end="")
                
                if episode_count % 100 == 0:
                    print(f'\r{print_str}')  # New line every 100 episodes
                
                # --- AUTOMATED EVALUATION & CHECKPOINTING ---
                if episode_count % evaluation_interval == 0 and episode_count > 0:
                    # Evaluate current policy at the current difficulty
                    eval_score = evaluate_policy(agent, num_eval_episodes=5, difficulty_stage=current_stage)
                    writer.add_scalar('Evaluation/Average_Score', eval_score, episode_count)
                    
                    # Save best model if it's the best score we've ever seen
                    if eval_score > best_score:
                        best_score = eval_score
                        best_episode = episode_count
                        torch.save(agent.qnetwork_local.state_dict(), f'best_model.pth')
                        print(f"\nNew best model saved with score {best_score:.2f} at episode {best_episode}")
                        
                    # --- CHANGE START ---
                    # Adaptive curriculum adjustment based on evaluation
                    if current_stage < max_stage and eval_score >= stage_thresholds[current_stage]:
                        current_stage += 1
                        parallel_envs.set_difficulty(current_stage)
                        print(f"\n--- Performance threshold met! Advancing to curriculum stage {current_stage} ---")
                        # Reset scores window to accurately reflect performance on the new difficulty
                        scores_window.clear()
                    # --- CHANGE END ---

                # --- PERIODIC SAVING ---
                episode_save_trigger = (episode_count % config.SAVE_INTERVAL_EPISODES == 0)
                time_save_trigger = (time.time() - last_save_time > config.SAVE_INTERVAL_SECONDS)
                
                if episode_save_trigger or time_save_trigger:
                    torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_parallel_{num_workers}workers.pth')
                    print(f"\n...Model saved at episode {episode_count}...")
                    if time_save_trigger:
                        last_save_time = time.time()
                
                # Reset this worker's metrics
                worker_episode_rewards[i] = 0.0
                worker_episode_steps[i] = 0
                worker_episode_passengers[i] = 0
                worker_episode_valid_actions[i] = []
                
                # Reset the environment for this worker
                new_state, new_valid_actions = parallel_envs.reset_worker(i)
                states[i] = new_state
                valid_actions_masks[i] = new_valid_actions
        
        # Check if we should stop training
        if episode_count >= config.N_EPISODES:
            break
    
    # Cleanup
    writer.close()
    parallel_envs.close()
    print(f"\nParallel training finished after {episode_count} episodes!")
    print(f"Total environment steps: {total_steps}")
    print(f"Steps per episode (avg): {total_steps / max(1, episode_count):.1f}")
    print(f"Final average score: {np.mean(scores_window):.2f}")


if __name__ == '__main__':
    # Number of parallel workers (adjust based on your CPU cores)
    # Typically use 2-8 workers for optimal performance
    num_parallel_workers = 2
    
    # Configuration
    train_config = TrainingConfig()
    rl_agent_config = ParallelRLAgentConfig(num_workers=num_parallel_workers)
    
    train_parallel(train_config, rl_agent_config, num_parallel_workers)