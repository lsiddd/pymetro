# ./train.py
import torch
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import os

from rl_agent.env import MiniMetroEnv
from rl_agent.dqn_agent import DQNAgent
from rl_agent.rl_config import RLAgentConfig, TrainingConfig

def train(config: TrainingConfig, agent_config: RLAgentConfig):
    """Deep Q-Learning training loop with enhanced logging and periodic saving."""
    
    # --- INICIALIZAÇÃO ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", f"minimetro_dqn_{timestamp}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Salva a configuração no diretório de log para reprodutibilidade
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        f.write(f"--- Training Config ---\n{str(config)}\n")
        f.write(f"--- Agent Config ---\n{str(agent_config)}\n")

    # Define headless=True para um treinamento muito mais rápido
    env = MiniMetroEnv(headless=True) 
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, config=agent_config, seed=0)

    scores_window = deque(maxlen=100)  # últimas 100 recompensas
    eps = config.EPS_START                    # inicializa o epsilon
    
    last_save_time = time.time()       # Inicializa o rastreador de tempo para salvar

    print("--- Starting Training ---")
    print(f"Agent will be saved every {config.SAVE_INTERVAL_EPISODES} episodes or {config.SAVE_INTERVAL_SECONDS / 60:.0f} minutes.")

    # --- LOOP DE TREINAMENTO ---
    for i_episode in range(1, config.N_EPISODES + 1):
        state, valid_actions_mask = env.reset()
        episode_reward = 0
        episode_losses = []
        passengers_delivered_this_episode = 0 # NOVO: Variável para armazenar os passageiros do episódio

        for t in range(config.MAX_T):
            action = agent.act(state, valid_actions_mask, eps)
            next_state, next_valid_actions_mask, reward_details, done, info = env.step(action)
            
            reward = reward_details['total']
            agent.step(state, action, reward, next_state, done)
            
            # Grava a perda se um passo de aprendizado ocorreu
            if agent.t_step == 0 and agent.last_loss is not None:
                episode_losses.append(agent.last_loss)
                
            state = next_state
            valid_actions_mask = next_valid_actions_mask
            episode_reward += reward
            
            if done:
                # Loga as estatísticas finais do jogo do dicionário 'info'
                writer.add_scalar('Game/Passengers Delivered', info.get('passengers_delivered', 0), i_episode)
                writer.add_scalar('Game/Final Score (from game)', info.get('final_score', 0), i_episode)
                writer.add_scalar('Game/Days Survived', info.get('days_survived', 0), i_episode)
                
                # NOVO: Captura o número de passageiros entregues para printar no console
                passengers_delivered_this_episode = info.get('passengers_delivered', 0)
                break 
        
        scores_window.append(episode_reward)
        eps = max(config.EPS_END, config.EPS_DECAY * eps)
        
        # --- LOGGING NO TENSORBOARD ---
        # Métricas de Desempenho
        writer.add_scalar('Performance/Episode Reward', episode_reward, i_episode)
        writer.add_scalar('Performance/Average Reward (100 episodes)', np.mean(scores_window), i_episode)
        
        # Componentes da Recompensa
        # (código de log da recompensa omitido por brevidade, mas permanece o mesmo)
        
        # Internos do Agente
        writer.add_scalar('Agent/Epsilon', eps, i_episode)
        writer.add_scalar('Agent/Episode Length (steps)', t + 1, i_episode)
        if episode_losses:
            writer.add_scalar('Agent/Average Loss', np.mean(episode_losses), i_episode)
        if agent.last_q_values is not None:
            writer.add_histogram('Agent/Q-Values', agent.last_q_values, i_episode)
        
        # Estado do Jogo
        writer.add_scalar('Game/Valid Actions Count', info.get('valid_actions_count', 0), i_episode)
        
        # --- SALVAMENTO PERIÓDICO ---
        episode_save_trigger = (i_episode % config.SAVE_INTERVAL_EPISODES == 0)
        time_save_trigger = (time.time() - last_save_time > config.SAVE_INTERVAL_SECONDS)

        # Printa o progresso no console
        # ATUALIZADO: Adicionado 'Passengers' ao print
        print_str = (
            f"Episode {i_episode}\t"
            f"Avg Reward: {np.mean(scores_window):.2f}\t"
            f"Passengers: {passengers_delivered_this_episode}\t"
            f"Epsilon: {eps:.4f}"
        )
        
        print(f'\r{print_str}', end="")
        
        if i_episode % 100 == 0:
            print(f'\r{print_str}') # Printa com uma nova linha a cada 100 episódios
            
        if episode_save_trigger or time_save_trigger:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            print(f"\n...Model saved at episode {i_episode}...")
            if time_save_trigger:
                last_save_time = time.time() # Reseta o timer após salvar
                
    writer.close()
    env.close()
    print("\nTraining finished.")

if __name__ == '__main__':
    train_config = TrainingConfig()
    rl_agent_config = RLAgentConfig()
    train(train_config, rl_agent_config)