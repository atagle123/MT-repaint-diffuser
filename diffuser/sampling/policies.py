from collections import namedtuple
import torch
import numpy as np
from diffuser.utils.arrays import atleast_2d
Trajectories = namedtuple('Trajectories', 'actions observations rewards task')


def expand_array(array, H, max_K=None):
    """
    Function to expand a NumPy array
    """
    max_K = max_K or H

    K, T = array.shape
    if max_K < K:
        # Create an array of zeros with shape (H - max_K, T)
        zeros = np.zeros((H - max_K, T), dtype=array.dtype)
        # Concatenate the last max_K rows of the original array with the zeros
        expanded_array = np.vstack((array[-max_K:, :], zeros))
    else:
        # Create an array of zeros with shape (H - K, T)
        zeros = np.zeros((H - K, T), dtype=array.dtype)
        # Concatenate the original array with the zeros
        expanded_array = np.vstack((array, zeros))
    
    return expanded_array


def expand_tensor(tensor, H, max_K=None):
    """
    Function to expand a tensor
    """
    max_K= max_K or H

    B, K, T = tensor.shape
    if max_K < K:
        zeros = torch.zeros(B,H - max_K, T, dtype=tensor.dtype, device=tensor.device)
                # Concatenar el tensor original con el tensor de ceros
        expanded_tensor = torch.cat((tensor[:,-max_K:,:], zeros), dim=1)
        
    else:
        # Crear un tensor de ceros con la forma (H-K, T)
        zeros = torch.zeros(B, H - K, T, dtype=tensor.dtype, device=tensor.device)
        # Concatenar el tensor original con el tensor de ceros
        expanded_tensor = torch.cat((tensor, zeros), dim=1)
    
    return expanded_tensor

def get_mask_from_tensor(tensor, H,observation_dim,max_K=None):
    """
    Function to get mask from a tensor
    """
    max_K= max_K or H
    #assert max_K>0
    B, K, T = tensor.shape
    ones = torch.ones(B, K, T, dtype=tensor.dtype, device=tensor.device)

    if max_K < K:
        zeros = torch.zeros(B, H - max_K+1, T, dtype=tensor.dtype, device=tensor.device)
                # Concatenar el tensor original con el tensor de ceros
        mask = torch.cat((ones[:,:(max_K-1),:], zeros), dim=1) # TODO REVISAR
        mask[:,max_K-1,:observation_dim]=1 # unmask the first state...
        
    else:
        # Crear un tensor de ceros con la forma (H-K, T)
        zeros = torch.zeros(B, H - K+1, T, dtype=tensor.dtype, device=tensor.device) # revisar
        # Concatenar el tensor original con el tensor de ceros
        mask = torch.cat((ones[:,:(K-1),:], zeros), dim=1)
        mask[:,K-1,:observation_dim]=1 # TODO revisar
    
    assert mask.shape==(B,H,T)

    return mask


def compute_reward_to_go_batch(rewards_batch, gamma):
    """
    Compute the reward-to-go for a batch of reward sequences with a discount factor gamma.
    
    Parameters:
        rewards_batch (torch.Tensor): A 2D tensor of shape (B, Horizon,1) where B is the batch size and Horizon is the length of each sequence.
        gamma (float): The discount factor.
    
    Returns:
        torch.Tensor: A 1D tensor of shape (B) containing reward-to-go values for each sequence in the batch.
    """

    assert rewards_batch.shape[2]==1

    rewards_batch=rewards_batch.squeeze(-1) # (B,H,1) -> (B,H)
    B, H = rewards_batch.shape

    gamma_tensor = torch.pow(gamma, torch.arange(H, dtype=torch.float32)).to(rewards_batch.device)
    gamma_matrix = gamma_tensor.unsqueeze(0).repeat(B, 1) # (B,H)

    # Apply gamma matrix to compute reward-to-go
    reward_to_go_batch = torch.sum(rewards_batch * gamma_matrix, dim=1)  # (B, H) -> (B)
    
    return reward_to_go_batch


def sort_by_values(actions, observations, rewards,gamma): # refactorizar funcion
    """
    [B,H,(A+S+R)]
    """
    values=compute_reward_to_go_batch(rewards,gamma) # (B,H,1)-> (B)

    inds = torch.argsort(values, descending=True)

    actions_sorted = actions[inds]
    observations_sorted=observations[inds]
    rewards_sorted=rewards[inds]
    values = values[inds]

    return actions_sorted,observations_sorted,rewards_sorted, values



class Policy:
    """
    Policy base class
    """

    def __init__(self, diffusion_model, dataset,gamma,resample_diff=1, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.resample_diff=resample_diff
        self.resample_counter=0
        self.sample_kwargs = sample_kwargs

    def __call__(self):
       raise NotImplementedError

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device
    
    def norm_evertything(self,trajectory,keys_order): # TODO test... maybe this has to be a method of dataset... check dims bewtween numpy and torch if it can generalize... 
        previous_dim=0
        for key in keys_order: # TODO construct dims one time only...
            current_dim=self.dataset.keys_dim_dict[key]+previous_dim
            unnormed_attribute = trajectory[:, :, previous_dim:current_dim]
            trajectory[:, :, previous_dim:current_dim]=self.dataset.normalizer.normalize(unnormed_attribute, key)
            previous_dim=current_dim
        return(trajectory)
    
    def unorm_everything(self,trajectory,keys_order): # TODO test...
        previous_dim=0
        for key in keys_order: # TODO construct dims one time only...
            current_dim=self.dataset.keys_dim_dict[key]+previous_dim
            unnormed_attribute = trajectory[:, :, previous_dim:current_dim]
            trajectory[:, :, previous_dim:current_dim]=self.dataset.normalizer.normalize(unnormed_attribute, key) 
            previous_dim=current_dim
        return(trajectory)
    
    def get_last_traj_rollout_torch(self,rollouts): # TODO maybe win time using the same policy class to not construct the torch rollout every time... 
        # K is the known history... 
        states_array,actions_array,rewards_array,total_reward_array,dones_array=rollouts.rollouts_to_numpy(index=-1)
        states_array=atleast_2d(states_array) # ensure that this arrays are at least 2d... 
        actions_array=atleast_2d(actions_array) 
        rewards_array=atleast_2d(rewards_array) 


        actions_array=expand_array(actions_array,H=states_array.shape[0]) # ensure that this arrays has the same dims of states,fill with zeros...  K+1,A
        rewards_array=expand_array(rewards_array,H=states_array.shape[0]) # K+1,1

        unkown_part=np.zeros((states_array.shape[0],self.task_dim)) # corresponds to the task and the reward to go... K+1, Task_dim+1
        known_trajectory=np.concatenate([states_array, actions_array ,rewards_array, unkown_part], axis=-1) # TODO this is the specific order... 

        known_trajectory_torch=torch.from_numpy(known_trajectory)  # K+1,T

        return (torch.unsqueeze(known_trajectory_torch, 0)) # 1,K+1,T

# idea: policy tiene metodos basicos y el metodo call depende del caso, por ejemplo en el exp rtg se llama a task inference con la historia ya conocida y tambien se llama a la eleccion de acciones... 

class Policy_repaint_return_conditioned(Policy):

    def __init__(self, diffusion_model, dataset,gamma,batch_size_sample,keys_order,resample_diff=1, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.batch_size_sample=batch_size_sample
        self.resample_diff=resample_diff
        self.resample_counter=0
        self.sample_kwargs = sample_kwargs
        self.horizon_sample=sample_kwargs.get("horizon_sample", self.dataset.horizon) # TODO test this... 
        self.keys_order=keys_order
        self.horizon=self.diffusion_model.horizon
    def __call__(self, rollouts,provide_task=None):
        """
        Main policy function that normalizes the data, calls the model and returns the result

        Args:
            conditions (torch.tensor): (B,H,T) a tensor filled with the known info and with 0 in everywhere else.
            mode (torch.tensor): (B,1) a tensor with the mode for each batch.
            verbose (bool): Print data

        Returns:
            df: dataframe with the data
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """ 
        if self.resample_counter==0:
            assert self.resample_diff>0
            self.resample_counter=self.resample_diff-1
            conditions=self.get_last_traj_rollout_torch(rollouts)  # conditions of dim 1,K+1,T
            normed_conditions=self.norm_evertything(conditions,keys_order=self.keys_order) # dim 1,K+1,T # TODO COMO HACER QUE no se normalicen las cosas que no se...maskeadas con 0...
            if provide_task is None:
                task=self.inpaint_task(normed_conditions,H=self.horizon_sample)
    #       print(" task",task)
            else:
                task=torch.from_numpy(provide_task)

            actions,observations,rewards, values=self.inpaint_action(normed_conditions,task,H=self.horizon_sample)
        else:
            self.resample_counter-=1
            actions=self.actions_plan
            observations=self.observations_plan
            rewards=self.rewards_plan
            task=self.task
        first_action=actions[0,0,:]
        self.actions_plan=actions[:,1:,:]
        self.observations_plan=observations[:,1:,:]
        self.rewards_plan=rewards[:,1:,:]
        self.task=task
     
        return(first_action.cpu().numpy(), Trajectories(actions,observations,rewards,task))

    
    def inpaint_task(self,normed_conditions,H):

        tensor_to_inpaint=expand_tensor(normed_conditions,H=H,max_K=H) # B,K,T-> B,H,T # TODO acordarse del ultimo state... TODO TEST THIS... with K> H

        mask=get_mask_from_tensor(normed_conditions,H=H,observation_dim=self.observation_dim,max_K=H) # mask #TODO TEST THIS... AND FIRST STATE...
        mask[:,:,-self.task_dim:]=0 # mask "known" task
        tensor_to_inpaint=mask*tensor_to_inpaint # ENSURE TO not have extra info TODO note that de igual manera esto va a pasar en el modelo...
        tensor_to_inpaint=tensor_to_inpaint.repeat(self.batch_size_sample,1,1).float() # ver que pasa si ya le pasamos un batch... 
        mask=mask.repeat(self.batch_size_sample,1,1).float()

        trajectories = self.diffusion_model(traj_known=tensor_to_inpaint, mask=mask,returns=None, **self.sample_kwargs) # notar que esto hace que inferir task dependa de returns inconditioned...
        task=self.infer_task_from_batch(trajectories.trajectories)
        return(task)
    
    def infer_task_from_batch(self,sampled_batch):
        task_mean=torch.mean(sampled_batch[:,:,-self.task_dim:],dim=(0,1))# the dims are specific for 2 dim goal task... 
        return(task_mean)
    
    def inpaint_action(self,normed_conditions,task,H):
        tensor_to_inpaint=expand_tensor(normed_conditions,H=H,max_K=1) # TODO acordarse del ultimo state... TODO TEST THIS... with K> H
        mask=get_mask_from_tensor(normed_conditions,H=H,observation_dim=self.observation_dim,max_K=1) # TODO TEST THIS...
        mask[:,:,-self.task_dim:]=1 # known task... 
        tensor_to_inpaint[:,:,-self.task_dim:]=task
        tensor_to_inpaint=mask*tensor_to_inpaint # ENSURE TO not have extra info TODO note that de igual manera esto va a pasar en el modelo...

        tensor_to_inpaint=tensor_to_inpaint.repeat(self.batch_size_sample,1,1).float()
        mask=mask.repeat(self.batch_size_sample,1,1).float()

        returns_batch = torch.ones(self.batch_size_sample, 1).float().to(device="cuda")

        trajectories = self.diffusion_model(traj_known=tensor_to_inpaint, mask=mask,returns=returns_batch, **self.sample_kwargs)

        return(self.infer_action_from_batch(trajectories.trajectories.clone()))
    
    def infer_action_from_batch(self,sampled_batch):
        unnormed_batch=self.unorm_everything(sampled_batch,keys_order=self.keys_order)

        observations=unnormed_batch[:,:,self.observation_dim:]
        actions=unnormed_batch[:,:,self.observation_dim:self.observation_dim+self.action_dim]
        rewards=unnormed_batch[:,:,self.observation_dim+self.action_dim:self.observation_dim+self.action_dim+1]

     #   actions,observations,rewards, values=sort_by_values(actions, observations, rewards,gamma=self.gamma) # in maze2d this doesnt make sense because the goal is to reach a objective...

        return(actions,observations,rewards, "_")