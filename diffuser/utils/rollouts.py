from dataclasses import dataclass, field
import numpy as np
import pickle
from typing import List, Any
from collections import namedtuple
from diffuser.utils.arrays import atleast_2d

Episode = namedtuple('episode',["states", "actions", "rewards", "total_reward", "dones"])

@dataclass
class Trajectory:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    total_reward: List[float] = field(default_factory=list)
    infos: List[Any] = field(default_factory=list)

@dataclass
class TrajectoryBuffer:
    trajectories: List[Trajectory] = field(default_factory=list)
    current_trajectory: Trajectory = field(default=None)

    def __init__(self, first_observation: np.ndarray, first_info: Any,action_dim: int):
        self.trajectories = []
        self.action_dim=action_dim
        self.start_trajectory(first_observation, first_info)

    def start_trajectory(self, first_observation: np.ndarray, first_info: Any):
        self.current_trajectory = Trajectory(
            states=[first_observation],
            actions=[],
            rewards=[],
            dones=[],
            total_reward=[],
            infos=[first_info]
        )

    def add_transition(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool, total_reward: float, info: Any):
        self.current_trajectory.states.append(state)
        self.current_trajectory.actions.append(action)
        self.current_trajectory.rewards.append(np.array([reward]))
        self.current_trajectory.dones.append(done)
        self.current_trajectory.total_reward.append(total_reward)
        self.current_trajectory.infos.append(info)

    def end_trajectory(self):
        self.trajectories.append(self.current_trajectory)
        self.current_trajectory = None


    def rollouts_to_numpy(self,index=-1):
        """
        Function to get the rollouts to numpy
        
        Args:
            index (int): index of the trajectory to select (-1 is the last) if there are no complete trajectory the current trajectory is passed.

        Returns:
            Episode (named tuple): named tuple containing arrays of the fields of the trajectory with the arrays with atleast 2 dims (H,T)
            
        """
        try:
            trajectory=self.trajectories[index]
        except:
            trajectory=self.current_trajectory

        states_array=np.stack(trajectory.states, axis=0) # H+1, state_dim
        try:
            actions_array=np.stack(trajectory.actions, axis=0) # H, action_dim
            rewards_array=np.stack(trajectory.rewards, axis=0) # H, 1
            total_reward_array=np.stack(trajectory.total_reward, axis=0) # H, 
            dones_array=np.stack(trajectory.dones, axis=0) # H, 

        except: # this exception happends in the first iteration... 
            actions_array=np.zeros((1,self.action_dim))
            rewards_array=np.zeros((1,1))
            total_reward_array=np.zeros((1,))
            dones_array=np.zeros((1,))

        states_array=atleast_2d(states_array) # ensure that this arrays are at least 2d... 
        actions_array=atleast_2d(actions_array) 
        rewards_array=atleast_2d(rewards_array) 
        total_reward_array=atleast_2d(total_reward_array)
        dones_array=atleast_2d(dones_array)

        return(Episode(states_array,actions_array,rewards_array,total_reward_array,dones_array))

    def save_trajectories(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectories, f)

    def load_trajectories(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.trajectories = pickle.load(f)
