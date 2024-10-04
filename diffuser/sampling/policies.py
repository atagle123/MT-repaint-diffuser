from collections import namedtuple
import torch
import numpy as np
from diffuser.utils.arrays import atleast_2d
from .sampling_utils import get_mask_from_tensor,expand_array,expand_tensor

Trajectories = namedtuple('Trajectories', 'actions observations rewards task')


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
    
    def get_last_traj_rollout_torch(self,rollouts): 
        # K is the known history... 
        trajectory=rollouts.rollouts_to_numpy(index="current")
        actions_array=trajectory.actions
        states_array=trajectory.states
        rewards_array=trajectory.rewards


        actions_array=expand_array(actions_array,H=states_array.shape[0]) # ensure that this arrays has the same dims of states,fill with zeros...  K+1,A TODO why?
        rewards_array=expand_array(rewards_array,H=states_array.shape[0]) # K+1,1

        unkown_part=np.zeros((states_array.shape[0],self.task_dim)) # corresponds to the task.. K+1, Task_dim
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