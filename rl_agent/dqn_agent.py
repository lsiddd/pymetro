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
    """Dueling Actor (Policy) Model."""
    def __init__(self, state_size, action_size, config, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared feature layers
        self.fc1 = nn.Linear(state_size, config.FC1_UNITS)
        self.fc2 = nn.Linear(config.FC1_UNITS, config.FC2_UNITS)

        # State-value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.FC2_UNITS, config.FC2_UNITS // 2),
            nn.ReLU(),
            nn.Linear(config.FC2_UNITS // 2, 1)
        )

        # Action-advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.FC2_UNITS, config.FC2_UNITS // 2),
            nn.ReLU(),
            nn.Linear(config.FC2_UNITS // 2, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        
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