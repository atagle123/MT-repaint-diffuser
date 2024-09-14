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

def get_mask_from_tensor(tensor, H,action_dim,observation_dim,max_K=None):
    """
    Function to get mask from a tensor
    """
    max_K= max_K or H
    B, K, T = tensor.shape
    ones = torch.ones(B, K, T, dtype=tensor.dtype, device=tensor.device)

    if max_K < K:
        zeros = torch.zeros(B, H - max_K+1, T, dtype=tensor.dtype, device=tensor.device)
                # Concatenar el tensor original con el tensor de ceros
        mask = torch.cat((ones[:,:(max_K-1),:], zeros), dim=1) # TODO REVISAR
        mask[:,max_K,action_dim:action_dim+observation_dim]=1 # unmask the first state...
        
    else:
        # Crear un tensor de ceros con la forma (H-K, T)
        zeros = torch.zeros(B, H - K+1, T, dtype=tensor.dtype, device=tensor.device) # revisar
        # Concatenar el tensor original con el tensor de ceros
        mask = torch.cat((ones[:,:(K-1),:], zeros), dim=1)
        mask[:,K-1,action_dim:action_dim+observation_dim]=1 # TODO revisar
    
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

    def __init__(self, diffusion_model, dataset,gamma, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
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

        unkown_part=np.zeros((states_array.shape[0],self.task_dim+1)) # corresponds to the task and the reward to go... K+1, Task_dim+1
        known_trajectory=np.concatenate([actions_array, states_array,rewards_array,unkown_part], axis=-1) # TODO this is the specific order... 

        known_trajectory_torch=torch.from_numpy(known_trajectory)  # K+1,T

        return (torch.unsqueeze(known_trajectory_torch, 0)) # 1,K+1,T

# idea: policy tiene metodos basicos y el metodo call depende del caso, por ejemplo en el exp rtg se llama a task inference con la historia ya conocida y tambien se llama a la eleccion de acciones... 


class Policy_mode(Policy): # TODO falta super init

    def __init__(self, diffusion_model, dataset,gamma, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.sample_kwargs = sample_kwargs
        self.inferred_task_list=[torch.tensor[0,0]] #?
        self.keys_order=("actions", "observations","rewards","returns","task")

    def __call__(self, rollouts):
        """
        Main policy function that normalizes the data, calls the model and returns the result

        Args:
            rollouts (Class): class that has trajectories history... 
            conditions (torch.tensor): (B,H,T) a tensor filled with the known info and with 0 in everywhere else.
            mode (torch.tensor): (B,1) a tensor with the mode for each batch.
            verbose (bool): Print data

        Returns:
            df: dataframe with the data
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """

        # 2. constructs the mask or the mode depending the case
        # batchifies the mode and the known batch
        # pass to the model
        # obtain the results
        # unnormalize, sort by rewards? and then pick the first action...
        # 

        conditions=self.get_last_traj_rollout_torch(rollouts)
        normed_conditions=self.norm_evertything(conditions,keys_order=self.keys_order) 
        mode_batch=self.create_mode(batch_size_sample=self.sample_kwargs.get("batch_size_sample"))
        conditions_batch=self.get_batch_from_conditions(conditions,batch_size_sample,horizon,infered_task)

        ## run reverse diffusion process
        trajectories = self.diffusion_model(traj_known=conditions, mode=mode, **self.sample_kwargs) # 

        trajectories=trajectories.trajectories

        # here get the action and the task... 
     #   print(trajectories[:, :, -self.task_dim:])
        #normed_rewards = trajectories[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1]
        #rewards = self.dataset.normalizer.unnormalize_torch(normed_rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        #normed_actions = trajectories[:, :, :self.action_dim]
        #actions = self.dataset.normalizer.unnormalize_torch(normed_actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

      #  normed_observations = trajectories[:, :, self.action_dim:self.action_dim+self.observation_dim]
      #  observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observation')

      #  normed_task = trajectories[:, :, -self.task_dim:] # TODO check... maybe do one function per sampling mode... 
       # task = self.dataset.normalizer.unnormalize_torch(normed_task, 'desired_goal')

        #TODO ver donde pponer el task sorted.. 
       # actions_sorted, observations_sorted, rewards_sorted, values = sort_by_values(actions, observations, rewards, gamma=self.gamma) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 
        # TODO ver que es mejor, si ordenar el rtg o los rewards... 
        ## extract first action
     #   action = actions_sorted[0, 0]

      #  trajectories = Trajectories(actions_sorted, observations_sorted, rewards_sorted,task) # generalizar aca ...

      #  return action, trajectories,task
        return(trajectories)


    def create_mode(self,batch_size_sample):
        # Create a tensor with the first B elements as 1 and the remaining B elements as 0
        mode_batch = torch.cat((torch.ones(batch_size_sample, 1), torch.zeros(batch_size_sample, 1)), dim=0)  # 2B,1 
        return mode_batch
    
    def get_batch_from_conditions(self,conditions,batch_size_sample,horizon,infered_task):
        # returns (2B,H,T)
        latest_state=conditions[-1,:] # latest state # assumes the rest is masked with zeros 
        action_inference=expand_tensor(latest_state, H=horizon, max_K=1) # (H,T)
        action_inference[:,-2:]=infered_task # set inferred task...
        action_inference_batch=action_inference.repeat(batch_size_sample,1,1) # B,H,T
        task_inference=expand_tensor(conditions, H=horizon) # H,T # TODO VER QUE PASA SI ES MENOR EL K QUE H...
        task_inference_batch=task_inference.repeat(batch_size_sample,1,1) # B,H,T
        batch=torch.cat((action_inference_batch, task_inference_batch), dim=0) # 2B,H,T TODO ver orden... 
        
        return batch


class Policy_repaint(Policy):

    def __init__(self, diffusion_model, dataset,gamma,batch_size_sample, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.dataset = dataset # dataset is a instance of the class sequence dataset
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.task_dim=diffusion_model.task_dim
        self.gamma=gamma
        self.batch_size_sample=batch_size_sample
        self.sample_kwargs = sample_kwargs
        self.keys_order=("actions", "observations","rewards","returns","task")
        self.horizon=self.diffusion_model.horizon
    def __call__(self, rollouts):
        """
        Main policy function that normalizes the data, calls the model and returns the result

        Args:
            conditions (torch.tensor): (B,H,T) a tensor filled with the known info and with 0 in everywhere else.
            mode (torch.tensor): (B,1) a tensor with the mode for each batch.
            verbose (bool): Print data

        Returns:
            df: dataframe with the data
        falta revisar que no inpaint step funcione, el sort tambien y la unnormalizacion tambien. evaluar todo aca.
        """ # first try: do everything at once, task inference and action inference
        conditions=self.get_last_traj_rollout_torch(rollouts)
       # print(conditions,conditions.shape)
        normed_conditions=self.norm_evertything(conditions,keys_order=self.keys_order) # TODO COMO HACER QUE no se normalicen las cosas que no se...maskeadas con 0...
       # print(normed_conditions.shape,normed_conditions)
        mask, batch_conditions=self.get_masks_and_batch(normed_conditions,batch_size_sample=self.batch_size_sample,max_K=16,H=self.horizon)
        # batchify masks and traj known... 
        trajectories = self.diffusion_model(traj_known=batch_conditions, mask=mask, **self.sample_kwargs) # 
        trajectories=trajectories.trajectories
        print(trajectories)

     #   print(trajectories[:, :, -self.task_dim:])
        #normed_rewards = trajectories[:, :, self.action_dim+self.observation_dim:self.action_dim+self.observation_dim+1]
        #rewards = self.dataset.normalizer.unnormalize_torch(normed_rewards, 'rewards') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 
        
        ## extract action [ batch_size x horizon x transition_dim + 1 ] transition_dim=actions+observations+rewards / + goal
        #normed_actions = trajectories[:, :, :self.action_dim]
        #actions = self.dataset.normalizer.unnormalize_torch(normed_actions, 'actions') # normalizar solo acciones y observaciones ... aunque tambien se podrian normalizar las rewards... probar esto... 

      #  normed_observations = trajectories[:, :, self.action_dim:self.action_dim+self.observation_dim]
      #  observations = self.dataset.normalizer.unnormalize_torch(normed_observations, 'observation')

      #  normed_task = trajectories[:, :, -self.task_dim:] # TODO check... maybe do one function per sampling mode... 
       # task = self.dataset.normalizer.unnormalize_torch(normed_task, 'desired_goal')

        #TODO ver donde pponer el task sorted.. 
       # actions_sorted, observations_sorted, rewards_sorted, values = sort_by_values(actions, observations, rewards, gamma=self.gamma) #sort by sampled returns. quizas esta no es la mejor metrica? puede ser que sea inconsistente? 
        # TODO ver que es mejor, si ordenar el rtg o los rewards... 
        ## extract first action
     #   action = actions_sorted[0, 0]

      #  trajectories = Trajectories(actions_sorted, observations_sorted, rewards_sorted,task) # generalizar aca ...

      #  return action, trajectories,task
        return(trajectories) 
    

    def get_masks_and_batch(self,normed_conditions,batch_size_sample,max_K,H): # TODO view a way to build the mask... possible masks: action inference, task inference, both together, previous + moving index of history... 
        # 1 option... everything at once...
        """
        normed_conditions (torch.Tensor): dim K,T
        """ # K, known history. IK inpainting Known history, H horizon
        K=normed_conditions.shape[0]
        
        tensor_to_inpaint=expand_tensor(normed_conditions,H=H,max_K=max_K) # TODO acordarse del ultimo state...

        mask=get_mask_from_tensor(normed_conditions, H,action_dim=self.action_dim,observation_dim=self.dataset.observation_dim,max_K=max_K)

        tensor_to_inpaint=tensor_to_inpaint*mask # TODO partial solution to masking the not known stuff
       # print(mask,mask.shape)
       # print(tensor_to_inpaint)
        return mask,tensor_to_inpaint