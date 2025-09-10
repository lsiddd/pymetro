import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import os

from rl_agent.parallel_env import ParallelEnvironments
from rl_agent.dqn_agent import DQNAgent
from rl_agent.rl_config import RLAgentConfig, TrainingConfig, ParallelRLAgentConfig


def train_parallel(config: TrainingConfig, agent_config: RLAgentConfig, num_workers: int = 4):
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
    parallel_envs = ParallelEnvironments(num_workers=num_workers)
    
    # Initialize agent
    agent = DQNAgent(
        state_size=parallel_envs.state_size, 
        action_size=parallel_envs.action_size, 
        config=agent_config, 
        seed=0
    )

    # Training metrics
    scores_window = deque(maxlen=100)
    eps = config.EPS_START
    last_save_time = time.time()
    
    # Episode tracking
    episode_count = 0
    total_steps = 0
    
    # Per-worker episode tracking
    worker_episode_rewards = [0.0] * num_workers
    worker_episode_steps = [0] * num_workers
    worker_episode_passengers = [0] * num_workers
    
    print("--- Starting Parallel Training ---")
    print(f"Agent will be saved every {config.SAVE_INTERVAL_EPISODES} episodes or {config.SAVE_INTERVAL_SECONDS / 60:.0f} minutes.")

    # Initialize all environments
    states, valid_actions_masks = parallel_envs.reset_all()
    
    # --- TRAINING LOOP ---
    while episode_count < config.N_EPISODES:
        # Get actions for all environments
        actions = agent.act_batch(states, valid_actions_masks, eps)
        
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
                scores_window.append(worker_episode_rewards[i])
                
                # --- TENSORBOARD LOGGING ---
                writer.add_scalar('Performance/Episode Reward', worker_episode_rewards[i], episode_count)
                writer.add_scalar('Performance/Average Reward (100 episodes)', np.mean(scores_window), episode_count)
                writer.add_scalar('Game/Passengers Delivered', final_passengers, episode_count)
                writer.add_scalar('Game/Final Score', final_score, episode_count)
                writer.add_scalar('Game/Days Survived', days_survived, episode_count)
                writer.add_scalar('Agent/Epsilon', eps, episode_count)
                writer.add_scalar('Agent/Episode Length (steps)', worker_episode_steps[i], episode_count)
                writer.add_scalar('Agent/Valid Actions Count', infos[i].get('valid_actions_count', 0), episode_count)
                
                # Log agent internals
                if agent.last_loss is not None:
                    writer.add_scalar('Agent/Last Loss', agent.last_loss, episode_count)
                if agent.last_q_values is not None:
                    writer.add_histogram('Agent/Q-Values', agent.last_q_values, episode_count)
                
                # --- CONSOLE LOGGING ---
                if episode_count % 10 == 0 or len(scores_window) <= 10:
                    print_str = (
                        f"Episode {episode_count:4d}\t"
                        f"Avg Reward: {np.mean(scores_window):7.2f}\t"
                        f"Passengers: {final_passengers:3d}\t"
                        f"Days: {days_survived:3d}\t"
                        f"Epsilon: {eps:.4f}\t"
                        f"Workers: {num_workers}"
                    )
                    print(f'\r{print_str}', end="")
                
                if episode_count % 100 == 0:
                    print(f'\r{print_str}')  # New line every 100 episodes
                
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
                
                # Reset the environment for this worker
                new_state, new_valid_actions = parallel_envs.reset_worker(i)
                states[i] = new_state
                valid_actions_masks[i] = new_valid_actions
                
                # Decay epsilon only when an episode completes
                eps = max(config.EPS_END, config.EPS_DECAY * eps)
        
        # Decay epsilon only when episodes complete, not every step
        # This was moved inside the episode completion logic above
        
        # Check if we should stop training
        if episode_count >= config.N_EPISODES:
            break
    
    # Cleanup
    writer.close()
    parallel_envs.close()
    print(f"\nParallel training finished after {episode_count} episodes!")
    print(f"Total environment steps: {total_steps}")
    print(f"Steps per episode (avg): {total_steps / max(1, episode_count):.1f}")
    print(f"Final average reward: {np.mean(scores_window):.2f}")


if __name__ == '__main__':
    # Number of parallel workers (adjust based on your CPU cores)
    # Typically use 2-8 workers for optimal performance
    num_parallel_workers = 4
    
    # Configuration
    train_config = TrainingConfig()
    rl_agent_config = ParallelRLAgentConfig(num_workers=num_parallel_workers)
    
    train_parallel(train_config, rl_agent_config, num_parallel_workers)