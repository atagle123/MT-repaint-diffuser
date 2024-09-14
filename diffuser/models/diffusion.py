from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
from tqdm import tqdm

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    Losses,
)

Sample = namedtuple('Sample', 'trajectories chains')


class GaussianDiffusion(nn.Module):
    """
    Base Gaussian diffusion model
    """
    def __init__(self,n_timesteps=20,
                 clip_denoised=True):
        super().__init__()

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))


    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            
        '''
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self):
        raise NotImplementedError
    
    def p_sample(self):
        raise NotImplementedError

    def p_sample_loop(self):
        raise NotImplementedError

    def conditional_sample(self):
        raise NotImplementedError

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample


    def p_losses(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError



"""
 def get_mask_loss_weights(self,K_step_mask):
        '''
            sets loss coefficients for masked trajectory
            
            K_step_mask: (B,)
        returns: loss_weights: (B,H,T)
        '''
        batch_size=len(K_step_mask)
        dim_weights=torch.ones(batch_size, self.horizon, self.transition_dim,dtype=torch.float32)
        #dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        #discounts = self.loss_discount ** torch.arange(self.horizon-K, dtype=torch.float) #TODO: horizon es todo el horizonte y h-K es lo que hay q predecir... 
        #discounts = discounts / discounts.mean() # TODO: para usar los discount podemos calcularlos antes... y sumarlos al mask 

        #loss_weights = torch.einsum('h-k,t->(h-k)t', discounts, dim_weights)

        loss_weights[:,K_step_mask, :self.action_dim] = self.action_weight
        loss_weights[:,K_step_mask, self.action_dim:self.action_dim+self.observation_dim] = 0 # because the initial state is given 
        loss_weights[:,:K_step_mask, :] = 0 # mask behind

        return loss_weights

"""



class GaussianDiffusion_task_rtg(GaussianDiffusion):
    """
    DDPM algorithm with 2 modes, task inference (unmasked plan) or policy sampling (masked plan and unmasked task) 
    """
    def __init__(self, model, horizon, observation_dim, action_dim,task_dim, n_timesteps=20,
        loss_type='l2', clip_denoised=True,
        action_weight=1.0,rtg_weight=1.0, loss_discount=1.0,p_mode=0.5):
        super().__init__(n_timesteps=n_timesteps,clip_denoised=clip_denoised)
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.task_dim=task_dim
        self.transition_dim = observation_dim + action_dim+1+task_dim+1 # (S+A+R+rtg+G)
        self.model = model
        self.action_weight=action_weight
        self.rtg_weight=rtg_weight
        self.loss_discount=loss_discount

        ## get loss coefficients and initialize objective
        self.bernoulli_dist = torch.distributions.Bernoulli(p_mode)

        self.loss_weights = self.get_loss_weights(action_weight,rtg_weight, loss_discount) # TODO 
        self.loss_fn = Losses[loss_type]()
        self.mask_dict=self.get_mask_mode()

    def get_loss_weights(self, action_weight, rtg_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t

        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight

        # always conditioning on s0
        loss_weights[0, self.action_dim:self.action_dim+self.observation_dim] = 0

        # manually set rtg weight
        loss_weights[0, -(self.task_dim+1)] = rtg_weight  # assumes A S R RTG TASK  
        return loss_weights.to(device="cuda").unsqueeze(0) # (1,H,T) TODO fix

    def get_mask_mode(self,device="cuda"): # assumes the following order: A S R RTG T 
        #TODO fix device 
        mask_dict={}

        # Crear el tensor de índices para la dimensión T
        indices = torch.arange(self.transition_dim).unsqueeze(0)  # Dimensiones (1, T)

        ### 0 mask ###

        mask_0 = (indices >= self.transition_dim-self.task_dim).float()  # (1, T)

        mask_0 = mask_0.expand(self.horizon, self.transition_dim).clone()  # (H, T)

        mask_0[0,self.action_dim:self.action_dim+self.observation_dim]= 1 # condition on s0
        mask_0=mask_0.to(device)
        mask_dict[0]=mask_0 # action inference 

        ### 1 mask ###

        # Comparar los índices con el umbral y crear la máscara
        mask_1 = (indices < self.transition_dim-self.task_dim).float()  # (1, T) -1 because the rtg is not a condition... TODO ver aca despues para el sampling si condicionar o no en el rtg..., no siempre esta disponible..  

        # Expandir la máscara a las dimensiones deseadas (H, T)
        mask_1 = mask_1.expand(self.horizon, self.transition_dim)  # (H, T)
        mask_1= mask_1.to(device)

        mask_dict[1]=mask_1 # task inference

        return(mask_dict)
    
    def get_mask_from_batch(self,mode_batch):
        stacked_values = torch.stack([self.mask_dict[0], self.mask_dict[1]])
        result_tensor = stacked_values[mode_batch]
        return(result_tensor)

    #------------------------------------------ sampling ------------------------------------------#


    @torch.no_grad()
    def p_mean_variance(self,x,t,mode_batch):
      #  if self.rtg_guidance: #TODO

       #     pass

        #else:
        t=t.clone().float().detach()

        epsilon = self.model(x=x, time=t,mode=mode_batch)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    
    @torch.no_grad()
    def p_sample(self, x, t,mode):
        b, *_, device = *x.shape, x.device
        batched_time = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_time,mode_batch=mode)

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0

        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return x_pred

    @torch.no_grad()
    def p_sample_loop(self, shape,traj_known, mode, disable_progess_bar=False, return_chain=False):
        """ 
        Classical DDPM sampling algorithm with masking and energy guidance

            Parameters:
                shape,
                traj_known: (B,H,T) 
                mode: assumes is in a batch of the way (B,1)
                disable_progess_bar=False, 
                return_chain=False

            Returns:
            sample (B,H,T) 
        
        """
        device = self.betas.device

        mask=self.get_mask_from_batch(mode) # (B,H,T) same dims as x 

        x = torch.randn(shape, device=device) # (B,H,T) same dims as x 
        x=traj_known*mask+x*(1-mask) 
        
        chain = [x] if return_chain else None  

        for t in tqdm(reversed(range(0, self.n_timesteps)), desc = 'sampling loop time step', total = self.n_timesteps,disable=disable_progess_bar):
            x = self.p_sample(x, t,mode)
            x=traj_known*mask+x*(1-mask)

            if return_chain: chain.append(x)

        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x,  chain)

    @torch.no_grad()
    def conditional_sample(self,traj_known, mode,  horizon_sample=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        horizon = horizon_sample or self.horizon
        shape = (len(traj_known), horizon, self.transition_dim)

        assert shape[0]==len(mode) # same batch size

        return self.p_sample_loop(shape,traj_known, mode, **sample_kwargs)
    

    def forward(self,traj_known,mode, *args, **kwargs):
        return self.conditional_sample(traj_known, mode, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_start, t):

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        mode_batch=self.bernoulli_dist.sample(sample_shape=(x_noisy.size(0), )).to(x_noisy.device).long() # (B,1) 0 o 1...
        mask=self.get_mask_from_batch(mode_batch) # (B,H,T) same dims as x 
 
        assert mask.shape==x_noisy.shape

        x_noisy=x_start*mask+x_noisy*(1-mask) # conditioning with mask...
        loss_weights=self.loss_weights*(1-mask)
 
        t = torch.tensor(t, dtype=torch.float)
        mode_batch=mode_batch.float().unsqueeze(-1)

        pred_epsilon = self.model(x_noisy,t,mode_batch)

        assert noise.shape == pred_epsilon.shape

        loss = self.loss_fn(pred_epsilon, noise,loss_weights=loss_weights)

        return loss
    

    def loss(self, x): 
    
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, t)
    



    # flavors1: raw, no condition on first state, action and test with added loss weight
    # 2 condition on first state and rtg
    # differiented reward function...
    # conditioned on returns
    # conditioned on mode ( exploration, task inference, planning) or: task planning and exploration


            ### opcion 2 ### faster... duplicate the batch... 
       #  mode=self.bernoulli_dist.sample(sample_shape=(x_noisy.size(0), 1)).to(x_noisy.device).float() # (B,1) 0 o 1...
        
      #  mask=get_mask_from_batch(mode_batch,mask_dict) (B,H,T) same dims as x 
        
    #    x_noisy=x_start*mask+x_noisy*(1-mask) # conditioning with mask... # TODO check this (works and the mask do what we want... )
        #noise=noise*(1-mask) # Check also this
        ############


class GaussianDiffusion_repaint_no_cond(GaussianDiffusion):
    """
    DDPM algorithm with repaint sampling
    """
    def __init__(self, model, horizon, observation_dim, action_dim,task_dim, n_timesteps=20,
        loss_type='l2', clip_denoised=True,
        action_weight=1.0,rtg_weight=1.0, loss_discount=1.0):
        super().__init__(n_timesteps=n_timesteps,clip_denoised=clip_denoised)
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.task_dim=task_dim
        self.transition_dim = observation_dim + action_dim+1+task_dim+1 # (S+A+R+G+rtg)
        self.model = model
        self.action_weight=action_weight
        self.rtg_weight=rtg_weight
        self.loss_discount=loss_discount

        ## get loss coefficients and initialize objective

        self.loss_weights = self.get_loss_weights(action_weight,rtg_weight, loss_discount) # TODO 
        self.loss_fn = Losses[loss_type]()

    def get_loss_weights(self, action_weight, rtg_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t

        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight

        # always conditioning on s0
        loss_weights[0, self.action_dim:self.action_dim+self.observation_dim] = 0

        # manually set rtg weight
        loss_weights[0, -(self.task_dim+1)] = rtg_weight  # assumes A S R RTG TASK  
        return loss_weights.to(device="cuda").unsqueeze(0) # (1,H,T) TODO fix

    #------------------------------------------ sampling ------------------------------------------#

    @torch.enable_grad()
    def p_mean_variance(self,x,t):
        t=t.clone().float().detach()
        epsilon = self.model(x=x, time=t)
        t = t.detach().to(torch.int64)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    
    @torch.inference_mode()
    def p_sample(self, x, t, traj_known, mask):
        b, *_, device = *x.shape, x.device

        if mask is not None:
            mask = mask.to(device)
            traj_known= traj_known.to(device)
            alpha_cumnprod_t = self.alphas_cumprod[t]
            gt_weight = torch.sqrt(alpha_cumnprod_t).to(device) 
            gt_part = gt_weight * traj_known
            noise_weight = torch.sqrt(1 - alpha_cumnprod_t).to(device)
            noise_part = noise_weight * torch.randn_like(x,device=device)
            weighed_gt = gt_part + noise_part
            x = (mask * weighed_gt) + ((1 - mask) * x)

        batched_time = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_time)

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise

        if t==0 and mask is not None:
            # if t == 0, we use the ground-truth image if in-painting
            x_pred = (mask * traj_known) +  ((1 - mask) * x_pred)

        return x_pred

    @torch.inference_mode()
    def p_sample_loop(self,
                shape,
                traj_known, # (B,H,T) same dims as x 
                mask, # (B,H,T) same dims as x 
                resample=True,
                resample_iter=10,
                resample_jump=3,
                resample_every=50,
                disable_progess_bar=False, 
                return_chain=False
                ):
        """ 
        Classical DDPM (check this) sampling algorithm with repaint sampling

            Parameters:
                shape:
                traj_known: (B,H,T) same dims as x 
                mask: (B,H,T) same dims as x 
                resample=True,
                resample_iter=10,
                resample_jump=3,
                resample_every=50,
                disable_progess_bar=False, 
                return_chain=False

            Returns:
            sample
        
        """

        device = self.betas.device

        x = torch.randn(shape, device=device)
        
        chain = [x] if return_chain else None

        for t in tqdm(reversed(range(0, self.n_timesteps)), desc = 'sampling loop time step', total = self.n_timesteps,disable=disable_progess_bar):
            x = self.p_sample(x=x, t=t, traj_known=traj_known, mask=mask)

            # Resampling loop: line 9 of Algorithm 1 in https://arxiv.org/pdf/2201.09865
            if resample is True and (t > 0) and (t % resample_every == 0 or t == 1) and mask is not None:
                # Jump back for resample_jump timesteps and resample_iter times
                for iter in tqdm(range(resample_iter), desc = 'resample loop', total = resample_iter,disable=disable_progess_bar):
                    t = resample_jump
                    beta = self.betas[t]
                    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * torch.randn_like(x)
                    for j in reversed(range(0, resample_jump)):
                        x = self.p_sample(x=x, t=t, traj_known=traj_known, mask=mask)

            if return_chain: chain.append(x)


        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x,  chain)

    @torch.inference_mode()
    def conditional_sample(self,
                traj_known,
                mask,
                horizon_sample=None,
                resample=True,
                resample_iter=10,
                resample_jump=3,
                resample_every=50,
                disable_progess_bar=False, 
                return_chain=False
                ):
        '''
            conditions : [ (time, state), ... ]
        '''
        horizon = horizon_sample or self.horizon
        
        batch_size=len(traj_known)

        assert batch_size==len(mask)

        return self.p_sample_loop(
                shape=(batch_size, horizon, self.transition_dim),
                traj_known=traj_known,
                mask=mask,
                resample=resample,
                resample_iter=resample_iter,
                resample_jump=resample_jump,
                resample_every=resample_every,
                disable_progess_bar=disable_progess_bar, 
                return_chain=return_chain)
    
    @torch.inference_mode()
    def forward(self,
                traj_known,
                mask,
                horizon_sample=None,
                resample=True,
                resample_iter=10,
                resample_jump=3,
                resample_every=50,
                disable_progess_bar=False, 
                return_chain=False
                ):
        
        return self.conditional_sample(
                traj_known=traj_known,
                mask=mask,
                horizon_sample=horizon_sample,
                resample=resample,
                resample_iter=resample_iter,
                resample_jump=resample_jump,
                resample_every=resample_every,
                disable_progess_bar=disable_progess_bar, 
                return_chain=return_chain)

    #------------------------------------------ training ------------------------------------------#

    def p_losses(self, x_start, t):

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
 
        t = torch.tensor(t, dtype=torch.float, requires_grad=True)

        x_noisy.requires_grad= True
        noise.requires_grad = True

        pred_epsilon = self.model(x_noisy,t)

        assert noise.shape == pred_epsilon.shape

        loss = self.loss_fn(pred_epsilon, noise,loss_weights=self.loss_weights) # Maybe do two functions to do that... 

        return loss
    

    def loss(self, x): 
    
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, t)
    



class GaussianDiffusion_repaint(GaussianDiffusion):
    """
    DDPM algorithm with repaint sampling
    """
    def __init__(self, model, horizon, observation_dim, action_dim,task_dim, n_timesteps=20,
        loss_type='l2', clip_denoised=True,
        action_weight=1.0,rtg_weight=1.0, loss_discount=1.0):
        super().__init__(n_timesteps=n_timesteps,clip_denoised=clip_denoised)
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.task_dim=task_dim
        self.transition_dim = observation_dim + action_dim+1+task_dim # (S+A+R+G)
        self.model = model
        self.action_weight=action_weight
        self.rtg_weight=rtg_weight
        self.loss_discount=loss_discount

        ## get loss coefficients and initialize objective

        self.loss_weights = self.get_loss_weights(action_weight,rtg_weight, loss_discount) # TODO 
        self.loss_fn = Losses[loss_type]()

    def get_loss_weights(self, action_weight, rtg_weight, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t

        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight

        # always conditioning on s0
        loss_weights[0, self.action_dim:self.action_dim+self.observation_dim] = 0

        # manually set rtg weight
        loss_weights[0, -(self.task_dim+1)] = rtg_weight  # assumes A S R RTG TASK  
        return loss_weights.to(device="cuda").unsqueeze(0) # (1,H,T) TODO fix

    #------------------------------------------ sampling ------------------------------------------#

    @torch.enable_grad()
    def p_mean_variance(self,x,t):
        t=t.clone().float().detach()
        epsilon = self.model(x=x, time=t)
        t = t.detach().to(torch.int64)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    
    @torch.inference_mode()
    def p_sample(self, x, t, traj_known, mask):
        b, *_, device = *x.shape, x.device

        if mask is not None:
            mask = mask.to(device)
            traj_known= traj_known.to(device)
            alpha_cumnprod_t = self.alphas_cumprod[t]
            gt_weight = torch.sqrt(alpha_cumnprod_t).to(device) 
            gt_part = gt_weight * traj_known
            noise_weight = torch.sqrt(1 - alpha_cumnprod_t).to(device)
            noise_part = noise_weight * torch.randn_like(x,device=device)
            weighed_gt = gt_part + noise_part
            x = (mask * weighed_gt) + ((1 - mask) * x)

        batched_time = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_time)

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise

        if t==0 and mask is not None:
            # if t == 0, we use the ground-truth image if in-painting
            x_pred = (mask * traj_known) +  ((1 - mask) * x_pred)

        return x_pred