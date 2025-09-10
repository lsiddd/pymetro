# ./rl_agent/rl_config.py
import torch

class RLAgentConfig:
    """
    Optimized configuration for stable DQN learning in complex Mini Metro environment.
    """
    def __init__(self):
        # --- Network Architecture ---
        # Increased network size for complex 302-dimensional state space
        self.FC1_UNITS = 256
        self.FC2_UNITS = 256
        
        # --- Learning Hyperparameters ---
        
        # Large replay buffer for diverse experiences
        self.BUFFER_SIZE = int(1e6)  # Increased to 1M for better sample diversity
        
        # Larger batch size for stable gradient updates
        self.BATCH_SIZE = 128
        
        # Standard discount factor
        self.GAMMA = 0.99
        
        # Target network soft update rate
        self.TAU = 5e-3  # Slightly faster target network updates
        
        # Lower learning rate for stable convergence
        self.LR = 5e-4   # Reduced from 1e-3 for better stability
        
        # Update frequency - train more frequently for faster learning
        # For parallel training, update less frequently since we collect more experiences per step
        self.UPDATE_EVERY = 4

        # --- Device ---
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

class TrainingConfig:
    """
    Optimized training configuration for faster convergence with balanced exploration.
    """
    def __init__(self):
        # Number of episodes for training
        self.N_EPISODES = 10000      # Increased for sufficient learning time
        
        # Maximum steps per episode
        self.MAX_T = 5000
        
        # Epsilon parameters for balanced exploration-exploitation
        self.EPS_START = 1.0         # Start with full exploration
        
        # Higher final epsilon to maintain some exploration
        self.EPS_END = 0.05          # Increased from 0.01 to maintain exploration
        
        # Faster epsilon decay for quicker transition to exploitation
        self.EPS_DECAY = 0.9995      # Faster decay than original 0.999
        
        # --- Saving Policy ---
        # More frequent saving to capture good checkpoints
        self.SAVE_INTERVAL_EPISODES = 200  # More frequent saves
        self.SAVE_INTERVAL_SECONDS = 600   # Save every 10 minutes

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])


class ParallelRLAgentConfig(RLAgentConfig):
    """
    Optimized configuration for parallel DQN training with multiple environments.
    """
    def __init__(self, num_workers=4):
        super().__init__()
        
        # Adjust batch size for parallel training
        # With multiple workers, we get more diverse experiences
        self.BATCH_SIZE = 64  # Reduced since we have more diverse data
        
        # Update frequency adjusted for parallel collection
        # With num_workers environments, we collect num_workers experiences per step
        self.UPDATE_EVERY = max(1, 4 // num_workers)  # Update more frequently with more workers
        
        # Larger replay buffer since we're collecting experiences faster
        self.BUFFER_SIZE = int(2e6)  # Increased for more experience diversity
        
        # Slightly faster target network updates for faster adaptation
        self.TAU = 1e-2