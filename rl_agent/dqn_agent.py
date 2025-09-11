# ./rl_agent/dqn_agent.py
from typing import Tuple, List, Optional, Any, Dict, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
import random
import math
from collections import namedtuple, deque

from rl_agent.rl_config import RLAgentConfig

# --- SumTree for Prioritized Experience Replay ---
class SumTree:
    """
    SumTree data structure for efficient priority-based sampling.
    This implementation provides O(log n) operations for updates and sampling.
    """
    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.tree: NDArray[np.float64] = np.zeros(2 * capacity - 1)
        self.data: NDArray[Any] = np.zeros(capacity, dtype=object)
        self.n_entries: int = 0
        self.pending_idx: int = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index based on priority sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return sum of all priorities."""
        return self.tree[0]

    def add(self, p: float, data: Any) -> None:
        """Add new experience with priority p."""
        idx = self.pending_idx + self.capacity - 1
        self.data[self.pending_idx] = data
        self.update(idx, p)

        self.pending_idx += 1
        if self.pending_idx >= self.capacity:
            self.pending_idx = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        """Update priority of experience at index idx."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get experience based on priority sum s."""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

# --- Noisy Linear Layer ---
class NoisyLinear(nn.Module):
    """Noisy Networks for Exploration - replaces standard Linear layer."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4) -> None:
        super(NoisyLinear, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.std_init: float = std_init

        # Learnable parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        """Reset noise for both weights and biases."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)

# Add curiosity module for better exploration
class CuriosityModule(nn.Module):
    def __init__(self, state_size: int, action_size: int, feature_dim: int = 256) -> None:
        super(CuriosityModule, self).__init__()
        self.feature_net: nn.Sequential = nn.Sequential(
            nn.Linear(state_size, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.inverse_net: nn.Sequential = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
        self.forward_net: nn.Sequential = nn.Sequential(
            nn.Linear(feature_dim + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def calculate_curiosity(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> float:
        state_feat = self.feature_net(state)
        next_state_feat = self.feature_net(next_state)
        
        # Forward model loss - prediction error
        # We detach the features to prevent the main network from being trained by the curiosity module
        predicted_next_feat = self.forward_net(torch.cat([state_feat.detach(), action], dim=1))
        forward_loss = F.mse_loss(predicted_next_feat, next_state_feat.detach())
        
        return forward_loss.item()


# --- Dueling QNetwork Model ---
class QNetwork(nn.Module):
    """Enhanced Dueling DQN with NoisyLinear layers for exploration."""
    def __init__(self, state_size: int, action_size: int, config: RLAgentConfig, seed: int) -> None:
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Shared feature layers - keep first layers regular for stability
        self.shared_fc1 = nn.Linear(state_size, config.FC1_UNITS)
        self.shared_fc2 = nn.Linear(config.FC1_UNITS, config.FC2_UNITS)
        self.shared_fc3 = nn.Linear(config.FC2_UNITS, config.FC2_UNITS)
        
        # Noisy layers for value stream
        self.value_fc1 = NoisyLinear(config.FC2_UNITS, config.FC2_UNITS // 2)
        self.value_fc2 = NoisyLinear(config.FC2_UNITS // 2, 1)

        # Noisy layers for advantage stream
        self.advantage_fc1 = NoisyLinear(config.FC2_UNITS, config.FC2_UNITS // 2)
        self.advantage_fc2 = NoisyLinear(config.FC2_UNITS // 2, action_size)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build a network that maps state -> action values."""
        # Shared feature extraction with regular layers
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout(x)
        x = F.relu(self.shared_fc3(x))
        
        # Value stream with noisy layers
        v = F.relu(self.value_fc1(x))
        v = self.dropout(v)
        V = self.value_fc2(v)
        
        # Advantage stream with noisy layers
        a = F.relu(self.advantage_fc1(x))
        a = self.dropout(a)
        A = self.advantage_fc2(a)
        
        # Combine V and A to get Q, ensuring identifiability
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
    
    def reset_noise(self) -> None:
        """Reset noise for all noisy layers."""
        self.value_fc1.reset_noise()
        self.value_fc2.reset_noise()
        self.advantage_fc1.reset_noise()
        self.advantage_fc2.reset_noise()

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using SumTree for efficient sampling.
    """
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int, device: torch.device, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001) -> None:
        self.action_size: int = action_size
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self.alpha: float = alpha  # Priority exponent
        self.beta: float = beta    # Importance sampling weight exponent
        self.beta_increment: float = beta_increment
        self.epsilon: float = 1e-6  # Small value to prevent zero priority
        
        self.tree: SumTree = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def _get_priority(self, error: Union[float, np.ndarray]) -> float:
        """Convert error to priority."""
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, state: NDArray[np.float32], action: int, reward: float, next_state: NDArray[np.float32], done: bool, error: Optional[float] = None) -> None:
        """Add experience to buffer with priority."""
        experience = self.experience(state, action, reward, next_state, done)
        
        # If no error is provided, use maximum priority to ensure it's sampled at least once
        if error is None:
            priority = self.tree.total() / self.buffer_size if self.tree.n_entries > 0 else 1.0
        else:
            priority = self._get_priority(error)
        
        self.tree.add(priority, experience)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
        """Sample batch with importance sampling weights."""
        batch: List[Any] = []
        indices: List[int] = []
        priorities: List[float] = []
        
        # Calculate priority segment
        priority_segment = self.tree.total() / self.batch_size
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(self.batch_size):
            # Sample from priority segments
            s = priority_segment * i + np.random.uniform(0, priority_segment)
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        priorities_array = np.array(priorities)
        min_prob = np.min(priorities_array) / self.tree.total()
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)
        weights = (priorities_array / self.tree.total() * self.tree.n_entries) ** (-self.beta)
        weights /= max_weight  # Normalize weights
        
        # Convert experiences to tensors
        states = torch.from_numpy(np.vstack([e.state for e in batch if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in batch if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in batch if e is not None]).astype(np.uint8)).float().to(self.device)
        weights_tensor = torch.from_numpy(weights).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones, indices, weights_tensor)

    def update_priorities(self, indices: List[int], errors: NDArray[np.float64]) -> None:
        """Update priorities of sampled experiences."""
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config: RLAgentConfig, seed: int = 0) -> None:
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.seed = random.seed(seed)
        self.config: RLAgentConfig = config
        self.device: torch.device = config.DEVICE

        # Q-Networks
        self.qnetwork_local: QNetwork = QNetwork(state_size, action_size, config, seed).to(self.device)
        self.qnetwork_target: QNetwork = QNetwork(state_size, action_size, config, seed).to(self.device)
        self.optimizer: torch.optim.Adam = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR)

        # Prioritized Replay memory
        self.memory: PrioritizedReplayBuffer = PrioritizedReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, seed, self.device)
        
        # Curiosity Module
        self.curiosity_module: CuriosityModule = CuriosityModule(state_size, action_size).to(self.device)
        
        # Internal state
        self.t_step: int = 0
        self.last_loss: Optional[float] = None
        self.last_q_values: Optional[NDArray[np.float32]] = None
        self.last_gradient_norm: float = 0

    def step(self, state: NDArray[np.float32], action: int, reward: float, next_state: NDArray[np.float32], done: bool) -> None:
        # Convert numpy inputs to tensors for the curiosity module
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        next_state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        # One-hot encode the action
        action_t = F.one_hot(torch.tensor([action], device=self.device), num_classes=self.action_size).float()

        # Add intrinsic curiosity reward
        with torch.no_grad(): # No gradients needed for curiosity calculation
            intrinsic_reward = self.curiosity_module.calculate_curiosity(state_t, action_t, next_state_t)
        total_reward = reward + intrinsic_reward * 0.1  # Scale intrinsic reward
        
        # Store experience with total reward
        self.memory.add(state, action, total_reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)
    
    def step_batch(self, experiences_list: List[Dict[str, Any]]) -> None:
        """Add multiple experiences to replay buffer and potentially learn."""
        # Add all experiences to memory with max priority
        for experience in experiences_list:
            state = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state = experience['next_state']
            done = experience['done']
            
            # Convert numpy inputs to tensors for the curiosity module
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            next_state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
            # One-hot encode the action
            action_t = F.one_hot(torch.tensor([action], device=self.device), num_classes=self.action_size).float()
            
            # Add intrinsic curiosity reward
            with torch.no_grad():
                 intrinsic_reward = self.curiosity_module.calculate_curiosity(state_t, action_t, next_state_t)
            total_reward = reward + intrinsic_reward * 0.1  # Scale intrinsic reward
            
            self.memory.add(state, action, total_reward, next_state, done)
        
        # Update step counter based on number of experiences
        self.t_step = (self.t_step + len(experiences_list)) % self.config.UPDATE_EVERY
        
        # Learn if we have enough experiences and it's time to update
        if self.t_step == 0 or len(experiences_list) >= self.config.UPDATE_EVERY:
            if len(self.memory) > self.config.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.config.GAMMA)

    def act(self, state: NDArray[np.float32], valid_actions_mask: Optional[NDArray[np.bool_]] = None, eps: float = 0.) -> int:
        """Act using noisy networks (eps parameter kept for compatibility but ignored)."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Reset noise for exploration
        self.qnetwork_local.reset_noise()
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Action selection with masking (no epsilon-greedy needed)
        if valid_actions_mask is not None:
            masked_action_values = action_values.clone()
            mask = torch.from_numpy(valid_actions_mask).bool().to(self.device)
            masked_action_values[0, ~mask] = -1e9
            self.last_q_values = masked_action_values.cpu().numpy()
            return np.argmax(masked_action_values.cpu().data.numpy())
        else:
            self.last_q_values = action_values.cpu().numpy()
            return np.argmax(action_values.cpu().data.numpy())
    
    def act_batch(self, states: NDArray[np.float32], valid_actions_masks: Optional[NDArray[np.bool_]] = None, eps: float = 0.) -> List[int]:
        """Get actions for multiple states simultaneously using noisy networks."""
        states_tensor = torch.from_numpy(states).float().to(self.device)
        batch_size = states_tensor.shape[0]
        
        # Reset noise for exploration
        self.qnetwork_local.reset_noise()
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(states_tensor)
        self.qnetwork_local.train()
        
        actions = []
        
        for i in range(batch_size):
            # Action selection with masking (no epsilon-greedy needed)
            if valid_actions_masks is not None:
                masked_action_values = action_values[i].clone()
                mask = torch.from_numpy(valid_actions_masks[i]).bool().to(self.device)
                masked_action_values[~mask] = -1e9
                action = np.argmax(masked_action_values.cpu().data.numpy())
            else:
                action = np.argmax(action_values[i].cpu().data.numpy())
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

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor], gamma: float) -> None:
        """Learn from prioritized experiences with importance sampling."""
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Reset noise for both networks
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()

        # Double DQN implementation
        # 1. Get action with maximum Q-value from local network for next state
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # 2. Get Q-value for that action from target network
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        
        # Calculate Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculate TD errors for priority updates
        td_errors = (Q_expected - Q_targets).detach().abs().cpu().numpy()
        
        # Calculate weighted loss using importance sampling weights
        elementwise_loss = F.mse_loss(Q_expected, Q_targets, reduction='none')
        loss = torch.mean(elementwise_loss * weights.unsqueeze(1))

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping and store norm for logging
        grad_norm = torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.config.GRADIENT_CLIP)
        if torch.is_tensor(grad_norm):
            self.last_gradient_norm = grad_norm.item()
        
        # Add learning rate scheduling based on stability
        if self.last_loss and self.last_loss > 1e-6 and abs(loss.item() - self.last_loss) / self.last_loss > 0.5:
            # Reduce learning rate if loss fluctuates too much
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(1e-6, param_group['lr'] * 0.9)
        
        self.last_loss = loss.item()

        self.optimizer.step()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.flatten())

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config.TAU)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)