# ./rl_agent/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

from rl_agent.rl_config import RLAgentConfig

# --- Dueling QNetwork Model ---
class QNetwork(nn.Module):
    """Enhanced Dueling DQN with deeper architecture and regularization."""
    def __init__(self, state_size, action_size, config, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Enhanced shared feature layers with dropout for regularization
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, config.FC1_UNITS),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.FC1_UNITS, config.FC2_UNITS),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Additional layer for complex state space
            nn.Linear(config.FC2_UNITS, config.FC2_UNITS),
            nn.ReLU()
        )

        # Enhanced state-value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.FC2_UNITS, config.FC2_UNITS // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.FC2_UNITS // 2, 1)
        )

        # Enhanced action-advantage stream  
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.FC2_UNITS, config.FC2_UNITS // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.FC2_UNITS // 2, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Shared feature extraction
        shared_features = self.shared_layers(state)
        
        # Separate value and advantage estimation
        V = self.value_stream(shared_features)
        A = self.advantage_stream(shared_features)
        
        # Combine V and A to get Q, ensuring identifiability
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size, config: RLAgentConfig, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.config = config
        self.device = config.DEVICE

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, config, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, config, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, seed, self.device)
        
        # Internal state
        self.t_step = 0
        self.last_loss = None
        self.last_q_values = None

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)
    
    def step_batch(self, experiences_list):
        """Add multiple experiences to replay buffer and potentially learn."""
        # Add all experiences to memory
        for experience in experiences_list:
            self.memory.add(
                experience['state'], 
                experience['action'], 
                experience['reward'], 
                experience['next_state'], 
                experience['done']
            )
        
        # Update step counter based on number of experiences
        self.t_step = (self.t_step + len(experiences_list)) % self.config.UPDATE_EVERY
        
        # Learn if we have enough experiences and it's time to update
        if self.t_step == 0 or len(experiences_list) >= self.config.UPDATE_EVERY:
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)

    def act(self, state, valid_actions_mask=None, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # --- Exploitation with Action Masking ---
            if valid_actions_mask is not None:
                masked_action_values = action_values.clone()
                mask = torch.from_numpy(valid_actions_mask).bool().to(self.device)
                
                # CORREÇÃO: Aplicar a máscara na primeira (e única) dimensão do batch
                # A forma do tensor é [1, 302] e a máscara é [302].
                # Indexar com [0, ~mask] alinha as formas corretamente.
                masked_action_values[0, ~mask] = -1e9
                
                self.last_q_values = masked_action_values.cpu().numpy() # Para logs
                return np.argmax(masked_action_values.cpu().data.numpy())
            else:
                self.last_q_values = action_values.cpu().numpy() # Para logs
                return np.argmax(action_values.cpu().data.numpy())
        else:
            # --- Exploração com Action Masking ---
            if valid_actions_mask is not None:
                valid_actions = np.where(valid_actions_mask)[0]
                if len(valid_actions) > 0:
                    return random.choice(valid_actions)
                else:
                    return 0 # Recorre a NO_OP se nenhuma ação for válida
            else:
                return random.choice(np.arange(self.action_size))
    
    def act_batch(self, states, valid_actions_masks=None, eps=0.):
        """Get actions for multiple states simultaneously."""
        states_tensor = torch.from_numpy(states).float().to(self.device)
        batch_size = states_tensor.shape[0]
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(states_tensor)
        self.qnetwork_local.train()
        
        actions = []
        
        for i in range(batch_size):
            if random.random() > eps:
                # Exploitation with action masking
                if valid_actions_masks is not None:
                    masked_action_values = action_values[i].clone()
                    mask = torch.from_numpy(valid_actions_masks[i]).bool().to(self.device)
                    masked_action_values[~mask] = -1e9
                    action = np.argmax(masked_action_values.cpu().data.numpy())
                else:
                    action = np.argmax(action_values[i].cpu().data.numpy())
                actions.append(action)
            else:
                # Exploration with action masking
                if valid_actions_masks is not None:
                    valid_actions = np.where(valid_actions_masks[i])[0]
                    if len(valid_actions) > 0:
                        action = random.choice(valid_actions)
                    else:
                        action = 0
                else:
                    action = random.choice(np.arange(self.action_size))
                actions.append(action)
        
        # Store Q-values for logging (from last state in batch)
        if valid_actions_masks is not None:
            masked_action_values = action_values[-1].clone()
            mask = torch.from_numpy(valid_actions_masks[-1]).bool().to(self.device)
            masked_action_values[~mask] = -1e9
            self.last_q_values = masked_action_values.cpu().numpy()
        else:
            self.last_q_values = action_values[-1].cpu().numpy()
        
        return actions

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # --- Implementação do Double DQN ---
        # 1. Pega a ação com o valor Q máximo da rede local para o próximo estado
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # 2. Pega o valor Q para essa ação da rede alvo
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        
        # Calcula os alvos Q para os estados atuais
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Pega os valores Q esperados do modelo local
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calcula a perda
        loss = F.mse_loss(Q_expected, Q_targets)
        self.last_loss = loss.item() # Armazena a perda para logs

        # Minimiza a perda
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Atualiza a rede alvo ---
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)