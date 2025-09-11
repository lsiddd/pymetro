# ./rl_agent/parallel_env.py

import multiprocessing as mp
import numpy as np
import torch
from typing import List, Tuple, Any, Optional
import pickle
import time

# This import is removed from here: from rl_agent.env import MiniMetroEnv
from rl_agent.dqn_agent import DQNAgent


class ParallelEnvironmentWorker:
    """Worker process that runs a single environment instance."""
    
    def __init__(self, worker_id: int, command_queue: mp.Queue, result_queue: mp.Queue):
        self.worker_id = worker_id
        self.command_queue = command_queue
        self.result_queue = result_queue
        self.env: Optional[Any] = None
        self.current_state: Optional[np.ndarray] = None
        self.current_valid_actions: Optional[np.ndarray] = None
    
    def run(self):
        """Main worker loop - processes commands until told to stop."""
        while True:
            try:
                command = self.command_queue.get(timeout=1.0)
                
                if command['type'] == 'init':
                    self._handle_init()
                elif command['type'] == 'reset':
                    # --- CHANGE START ---
                    self._handle_reset(command['difficulty_stage'])
                    # --- CHANGE END ---
                elif command['type'] == 'step':
                    self._handle_step(command['action'])
                elif command['type'] == 'close':
                    self._handle_close()
                    break
                    
            except Exception as e:
                self.result_queue.put({
                    'worker_id': self.worker_id,
                    'type': 'error',
                    'error': str(e)
                })
    
    def _handle_init(self):
        """Initialize the environment."""
        # The import is moved here, so it only happens inside the worker process
        from rl_agent.env import MiniMetroEnv
        self.env = MiniMetroEnv(headless=True)
        self.result_queue.put({
            'worker_id': self.worker_id,
            'type': 'init_done',
            'state_size': self.env.state_size,
            'action_size': self.env.action_size
        })
    
    # --- CHANGE START ---
    def _handle_reset(self, difficulty_stage):
        """Reset the environment and return initial state."""
        state, valid_actions = self.env.reset(difficulty_stage=difficulty_stage)
    # --- CHANGE END ---
        self.current_state = state
        self.current_valid_actions = valid_actions
        
        self.result_queue.put({
            'worker_id': self.worker_id,
            'type': 'reset_done',
            'state': state,
            'valid_actions': valid_actions
        })
    
    def _handle_step(self, action):
        """Execute one step and return results."""
        next_state, next_valid_actions, reward_details, done, info = self.env.step(action)
        
        # Store experience tuple for the agent
        experience = {
            'state': self.current_state,
            'action': action,
            'reward': reward_details['total'],
            'next_state': next_state,
            'done': done
        }
        
        self.current_state = next_state
        self.current_valid_actions = next_valid_actions
        
        self.result_queue.put({
            'worker_id': self.worker_id,
            'type': 'step_done',
            'experience': experience,
            'next_state': next_state,
            'next_valid_actions': next_valid_actions,
            'reward_details': reward_details,
            'done': done,
            'info': info
        })
    
    def _handle_close(self):
        """Clean up and close the environment."""
        if self.env:
            self.env.close()
        self.result_queue.put({
            'worker_id': self.worker_id,
            'type': 'close_done'
        })


class ParallelEnvironments:
    """Manages multiple environment instances running in parallel processes."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers: List[ParallelEnvironmentWorker] = []
        self.command_queues: List[mp.Queue] = []
        self.result_queue: mp.Queue = mp.Queue()
        self.processes: List[mp.Process] = []
        
        # Environment info (will be set after initialization)
        self.state_size: Optional[int] = None
        self.action_size: Optional[int] = None
        
        # Current states for each worker
        self.worker_states: List[Optional[np.ndarray]] = [None] * num_workers
        self.worker_valid_actions: List[Optional[np.ndarray]] = [None] * num_workers
        
        # --- CHANGE START ---
        self.current_difficulty_stage: int = 0
        # --- CHANGE END ---

        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize all worker processes."""
        for i in range(self.num_workers):
            command_queue = mp.Queue()
            worker = ParallelEnvironmentWorker(i, command_queue, self.result_queue)
            
            # Start the worker process
            process = mp.Process(target=worker.run)
            process.start()
            
            self.command_queues.append(command_queue)
            self.processes.append(process)
            
            # Send initialization command
            command_queue.put({'type': 'init'})
        
        # Wait for all workers to initialize
        initialized = 0
        start_time = time.time()
        timeout = 60  # 60 seconds timeout
        
        while initialized < self.num_workers:
            try:
                result = self.result_queue.get(timeout=5.0)  # 5 second timeout per get
                if result['type'] == 'init_done':
                    if self.state_size is None:
                        self.state_size = result['state_size']
                        self.action_size = result['action_size']
                    initialized += 1
                elif result['type'] == 'error':
                    raise RuntimeError(f"Worker initialization failed: {result.get('error', 'Unknown error')}")
            except:
                if time.time() - start_time > timeout:
                    raise RuntimeError(f"Timeout waiting for worker initialization. Only {initialized}/{self.num_workers} workers initialized.")
                continue
    
    # --- CHANGE START ---
    def set_difficulty(self, stage: int):
        """Set the difficulty for all subsequent environment resets."""
        self.current_difficulty_stage = stage

    def reset_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments and return initial states."""
        # Send reset commands to all workers
        for queue in self.command_queues:
            queue.put({'type': 'reset', 'difficulty_stage': self.current_difficulty_stage})
    # --- CHANGE END ---
        
        # Collect results
        states = []
        valid_actions = []
        reset_count = 0
        
        while reset_count < self.num_workers:
            result = self.result_queue.get()
            if result['type'] == 'reset_done':
                worker_id = result['worker_id']
                self.worker_states[worker_id] = result['state']
                self.worker_valid_actions[worker_id] = result['valid_actions']
                
                states.append(result['state'])
                valid_actions.append(result['valid_actions'])
                reset_count += 1
        
        return np.array(states), np.array(valid_actions)
    
    def step_all(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, List[Any], List[bool], List[Any], List[Any]]:
        """Execute actions in all environments."""
        assert len(actions) == self.num_workers, f"Expected {self.num_workers} actions, got {len(actions)}"
        
        # Send step commands to all workers
        for i, action in enumerate(actions):
            self.command_queues[i].put({'type': 'step', 'action': action})
        
        # Collect results
        experiences = []
        next_states = []
        next_valid_actions = []
        reward_details_list = []
        dones = []
        infos = []
        
        step_count = 0
        while step_count < self.num_workers:
            result = self.result_queue.get()
            if result['type'] == 'step_done':
                worker_id = result['worker_id']
                
                # Store experience for replay buffer
                experiences.append(result['experience'])
                
                # Update worker states
                self.worker_states[worker_id] = result['next_state']
                self.worker_valid_actions[worker_id] = result['next_valid_actions']
                
                # Collect return values
                next_states.append(result['next_state'])
                next_valid_actions.append(result['next_valid_actions'])
                reward_details_list.append(result['reward_details'])
                dones.append(result['done'])
                infos.append(result['info'])
                
                step_count += 1
        
        return (np.array(next_states), np.array(next_valid_actions), 
                reward_details_list, dones, infos, experiences)
    
    # --- CHANGE START ---
    def reset_worker(self, worker_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Reset a specific worker environment."""
        self.command_queues[worker_id].put({'type': 'reset', 'difficulty_stage': self.current_difficulty_stage})
    # --- CHANGE END ---
        
        # Wait for reset to complete
        while True:
            result = self.result_queue.get()
            if result['type'] == 'reset_done' and result['worker_id'] == worker_id:
                self.worker_states[worker_id] = result['state']
                self.worker_valid_actions[worker_id] = result['valid_actions']
                return result['state'], result['valid_actions']
    
    def get_current_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current states and valid actions for all workers."""
        states = [state for state in self.worker_states if state is not None]
        valid_actions = [actions for actions in self.worker_valid_actions if actions is not None]
        return np.array(states), np.array(valid_actions)
    
    def close(self):
        """Close all environments and terminate worker processes."""
        # Send close commands
        for queue in self.command_queues:
            queue.put({'type': 'close'})
        
        # Wait for confirmation
        closed_count = 0
        while closed_count < self.num_workers:
            try:
                result = self.result_queue.get(timeout=2.0)
                if result['type'] == 'close_done':
                    closed_count += 1
            except:
                break
        
        # Terminate processes
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()