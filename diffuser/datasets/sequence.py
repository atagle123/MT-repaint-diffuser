from collections import namedtuple
import numpy as np
import torch
import gymnasium as gym
import minari
from diffuser.utils.config import import_class
from diffuser.utils.arrays import atleast_2d,pad, pad_min
from .auxiliar_functions import find_key_in_data


Batch = namedtuple('Batch', 'trajectories')
RewardBatch = namedtuple('Batch', 'trajectories returns')
TaskRewardBatch=namedtuple('Batch', 'trajectories returns task')


class SequenceDataset(torch.utils.data.Dataset):
    """
    Base class to make a sequence dataset from minari
    """

    def __init__(self, 
                 dataset_name="D4RL/pointmaze/umaze-dense-v2", 
                 horizon=64,
                 normalizer="datasets.normalization.GaussianNormalizer", 
                 max_path_length=1000,
                 max_n_episodes=100000, 
                 termination_penalty=0,
                 seed=None,
                 use_padding=True,
                 normed_keys=[ "observations", 'actions','rewards',"task"],
                 view_keys_dict={"observations":"observation","actions":"actions","rewards":"rewards","task":"desired_goal"}, # the name of the attribute vs the name we want in the dataset.
                 discount=0.99
                 ):
        """
        Initializes the class with the specified parameters.

        Args:

            dataset_name (str): The name of the dataset to be used. Should be downloaded local minari.
            horizon (int): The horizon for episodes, indicating the maximum number of steps in an episode. Defaults to 64.
            normalizer (str): The path to the normalizer class to be used for data normalization.
            max_path_length (int): The maximum length of paths (episodes) to be considered. Defaults to 1000.
            max_n_episodes (int): The maximum number of episodes to consider. Defaults to 100000.           
            termination_penalty (float): A penalty applied at the end of episodes to encourage early termination. Defaults to 0.            
            seed (int or None): Random seed for reproducibility. If None, a random seed will be generated.            
            use_padding (bool): Whether to use padding in the dataset. Defaults to True.            
            normed_keys (list of str): A list of keys for which normalization is to be applied. Defaults to ["observations", 'actions', 'rewards', "task"].            
            view_keys_dict (dict): A mapping from attribute names to the names that should be used in the dataset. Defaults to {"observations": "observation", "actions": "actions", "rewards": "rewards", "task": "desired_goal"}.           
            discount (float): The discount factor for future rewards, between 0 and 1. Defaults to 0.99.
        """
        
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.termination_penalty=termination_penalty
        self.use_padding=use_padding
        self.discount=discount
        self.normed_keys=normed_keys
        self.view_keys_dict=view_keys_dict
        self.seed=seed
        self.dataset_name=dataset_name
        self.max_n_episodes=max_n_episodes

        self.get_env_attributes()

        self.make_dataset(view_keys_dict=view_keys_dict)
        self.make_indices(horizon)

        self.make_returns()

        self.normalize_dataset(normed_keys=self.normed_keys,normalizer=import_class(normalizer))
        self.get_norm_keys_dim()

        #self.sanity_test()


    def get_env_attributes(self):
        """
        Function to to get the env and his attributes from the dataset_name

        """
        self.minari_dataset=minari.load_dataset(self.dataset_name)
        self.minari_dataset.set_seed(seed=self.seed)
        self.env = self.minari_dataset.recover_environment()
        action_space=self.env.action_space
        observation_space = self.env.observation_space

        assert self.minari_dataset.total_episodes<= self.max_n_episodes

        self.n_episodes=self.minari_dataset.total_episodes

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]

        
    def make_dataset(self,view_keys_dict):
        """
        Transforms minari dataset to a general format.

        Args: 
            view_keys_dict (dict): A mapping from attribute names to the names that should be used in the dataset. Also is all the information to be processed in the dataset.
        
        Returns a episodes dict with this format: 
            episodes_dict.keys-> ["observations","actions","rewards","terminations","truncations","total_returns"]
            episodes_dict.values-> np.array 2d [H,Dim]
        """ 
        print("Making dataset... ")

        episodes_generator = self.minari_dataset.iterate_episodes()
        self.episodes={}

        ### generate new dataset in the format ###
        for episode in episodes_generator:
            dict={}
            for new_name_key,key in view_keys_dict.items():
                
                attribute=find_key_in_data(episode,key)

                if attribute is None:
                    raise KeyError(f" Couldn't find a np.array value for the key {key}")

                attribute_2d=atleast_2d(attribute)

                if self.use_padding:
                    attribute=pad(attribute_2d,max_len=self.max_path_length)

                    assert attribute.shape==(self.max_path_length,attribute_2d.shape[-1])

                else:
                    attribute=pad_min(attribute_2d,min_len=self.horizon)

                if key=="rewards":  
                    if episode.terminations.any():
                        episode_lenght=episode.total_timesteps
                        attribute[episode_lenght-1]+=self.termination_penalty  # o quizas -1 tambien sirve...
           
                dict[new_name_key]=attribute

            self.episodes[episode.id]=dict


    def normalize_dataset(self,normed_keys,normalizer):
        """
        Function to normalize the dataset

        Args: 
            normed_keys (list): A list of the keys od the dataset to normalize.
            normalizer (class instance): instanfe of a normalization class with normalization functions.
        
        Returns a episodes dict with normalized fields.
        """ 
        print("Normalizing dataset... ")
        
        self.normalizer=normalizer(dataset=self.episodes,normed_keys=normed_keys)

        for ep_id, dict in self.episodes.items():
            for key,attribute in dict.items():

                if key in normed_keys:
                    attribute=self.normalizer.normalize(attribute,key) # normalize

                    dict[key]=attribute

            self.episodes[ep_id]=dict


    def get_norm_keys_dim(self):
        """
        Function to get a dictionary with the keys and his dim.
        """
        self.keys_dim_dict={}
        dict=self.episodes[0]
        for key,attribute in dict.items():
            self.keys_dim_dict[key]=attribute.shape[-1]


    def make_indices(self, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        print("Making indices... ")
        indices = []
        
        for ep_id,episode_dict in self.episodes.items():  # assumes padding fix it to use no padding
            
            episode_lenght=len(episode_dict["actions"])

            assert self.max_path_length>=episode_lenght
            
            max_start = min(episode_lenght - 1, self.max_path_length - horizon)

            if not self.use_padding:

                max_start = min(episode_lenght - horizon,self.max_path_length - horizon) # assumes thath the min lenght is horizon... 
                
                assert episode_lenght>=horizon

            for start in range(max_start+1):
                end = start + horizon
                indices.append((ep_id, start, end))

        indices = np.array(indices)
        self.indices=indices

    
    def inference_mode(self):
        del self.episodes; del self.indices #save memory in inference 


    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, idx):
        ep_id, start, end = self.indices[idx]
        episode=self.episodes[ep_id]

        keys_list=[]
        for key,dim in self.keys_dim_dict.items():
            keys_list.append(episode[key][start:end])

        trajectories = np.concatenate(keys_list, axis=-1)

        batch = Batch(trajectories)

        return batch


    def __getstate__(self):
        return self.__dict__
    
    
    def __setstate__(self, d):
     print("I'm being unpickled with these values: " + repr(d))
     self.__dict__ = d
    

    def make_returns(self): # TODO tiene mas sentido hacerlo usando los indices... problema: make returns despues de hacer el dataset sesga a que hayan muchos 0s como rewards
        """
        Function to make returns, it considers the normalized reward to go of the current state of a trajectory.
        Later it creates a field in the episodes dict with returns, and adds returns to the keys to normalize.
        """
        print("Making returns... ")
        
        discount_array=self.discount ** np.arange(self.max_path_length) # (H)
        discount_array=atleast_2d(discount_array)

        norm_factors=[self.calc_norm_factor(self.discount,horizon) for horizon in range(self.max_path_length)] # ordered list with list[horizon]-> norm_factor(horizon)

        for ep_id, dict in self.episodes.items():
            rtg_list=[]
            rewards=dict["rewards"]
            horizon=len(rewards)-1  # the -1 is correct
            rtg_partial=np.sum(rewards*discount_array[:(horizon+1)]) # (H)*(H)-> 1
            rtg_list.append(rtg_partial*norm_factors[horizon])

            for rew in rewards:
                rtg_partial=(rtg_partial-rew[0])/self.discount 
                horizon-=1

                rtg_list.append(rtg_partial*norm_factors[horizon])
            
            returns_array=np.array(rtg_list[:-1],dtype=np.float32) 
            assert returns_array.shape[0]==rewards.shape[0]

            self.episodes[ep_id]["returns"]=atleast_2d(returns_array)

        self.normed_keys.append("returns")
    
    def calc_norm_factor(self,discount,horizon):
        """
        Function to calculate norm factor
        """
        norm_factor=(1-discount)/(1-discount**(horizon+1))
        return(norm_factor)


    def sanity_test(self):
        raise  NotImplementedError
        #check dims... 


class Maze2d_inpaint_dataset(SequenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_env_attributes(self):

        self.minari_dataset=minari.load_dataset(self.dataset_name)
        self.minari_dataset.set_seed(seed=self.seed)
        self.env = self.minari_dataset.recover_environment()
        action_space=self.env.action_space
        observation_space = self.env.observation_space["observation"]

        assert self.minari_dataset.total_episodes<= self.max_n_episodes

        self.n_episodes=self.minari_dataset.total_episodes

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n

        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]

        self.observation_dim = observation_space.shape[0]


    def make_dataset(self,view_keys_dict):
        """
        Transforms minari dataset to a standard way... 

        Format: episodes_dict.keys-> ["observations","actions","rewards","terminations","truncations","total_returns"]
                episodes_dict.values-> np.array 2d [H,Dim]
        """ 
        print("Making dataset... ")

        episodes_generator = self.minari_dataset.iterate_episodes()
        self.episodes={}
        ### generate new dataset in the format ###
        for episode in episodes_generator:
            dict={}
            for new_name_key,key in view_keys_dict.items():
                
                attribute=find_key_in_data(episode,key)

                if attribute is None:
                    raise KeyError(f" Couldn't find a np.array value for the key {key}")

                attribute_2d=atleast_2d(attribute)

                ###
                # specific truncation in maze2d dataset... 2 options truncate first element or change desired goal... 
                ###
                attribute_2d=attribute_2d[1:,:]

                if self.use_padding:
                    attribute=pad(attribute_2d,max_len=self.max_path_length)

                    assert attribute.shape==(self.max_path_length,attribute_2d.shape[-1])

                else:
                    attribute=pad_min(attribute_2d,min_len=self.horizon)

                if key=="rewards":  
                    if episode.terminations.any():
                        episode_lenght=episode.total_timesteps
                        attribute[episode_lenght-1]+=self.termination_penalty  # o quizas -1 tambien sirve...
                        
                dict[new_name_key]=attribute
            self.episodes[episode.id]=dict


    def __getitem__(self, idx):
        ep_id, start, end = self.indices[idx]
        episode=self.episodes[ep_id]

        observations = episode['observations'][start:end]
        actions = episode['actions'][start:end]
        rewards=episode['rewards'][start:end]
        returns=episode["returns"][start:end]
        task=episode["task"][start:end]

        trajectories = np.concatenate([actions, observations,rewards,returns,task], axis=-1)

        batch = Batch(trajectories)

        return batch
    

class Maze2d_inpaint_dataset_returns(Maze2d_inpaint_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def make_returns(self):
        """
        Specific return making for maze2d and goal reaching targets.
        """
        print("Making returns... ")
        
        discount_array=self.discount ** np.arange(self.max_path_length) # (H)
        discount_array=atleast_2d(discount_array)

        for ep_id, dict in self.episodes.items():
            rewards=dict["rewards"]

            horizon=len(rewards)
            optimal=self.is_optimal_episode(dict["observations"],dict["task"]) # TODO check this...
            #int(optimal)*discount_array[horizon]
            returns_array = np.full((horizon, 1), int(optimal),dtype=np.float32)

            assert returns_array.shape[0]==rewards.shape[0]

            self.episodes[ep_id]["returns"]=atleast_2d(returns_array)

        self.normed_keys.append("returns") #no need to norm in this setup

    def is_optimal_episode(self,state,goal):
        """
        Function to check if a trajectory in maze2d is optimal.
        Note that the constraint to be optimal is that the euclidean distance between the goal and the position is less than 0.5m.
        """
        distances = np.linalg.norm(state[:,:2] - goal, axis=1)
        if distances[-1]<=0.5: #or distances[-2]<=0.5:
            return(True)
        else:
            return(False)


    def __getitem__(self, idx):
        ep_id, start, end = self.indices[idx]
        episode=self.episodes[ep_id]

        observations = episode['observations'][start:end]
        actions = episode['actions'][start:end]
        rewards=episode['rewards'][start:end]
        returns=episode["returns"][start]
        task=episode["task"][start:end]

        trajectories = np.concatenate([observations, actions, rewards, task], axis=-1)
        
        batch = RewardBatch(trajectories,returns)

        return batch