import torch
import numpy as np



def get_mask_from_tensor(B, K, T, H, observation_dim, dtype=torch.float32, device="cuda"):
    """
    Function to get mask from a tensor
    Args:
        B (int): Batch size
        K (int): Large of unmasked tensor
        T (int): Transition dim
        H (int): Desired horizon
        observation_dim (int): obs dim
        dtype (dtype): dtype
        device (str): device

    Returns:
        A mask with the desired attributes. 

    """
    ones = torch.ones(B, K, T, dtype=dtype, device=device)

    # Crear un tensor de ceros con la forma (H-K, T)
    zeros = torch.zeros(B, H - K+1, T, dtype=dtype, device=device) # revisar
    # Concatenar el tensor original con el tensor de ceros
    mask = torch.cat((ones[:,:(K-1),:], zeros), dim=1)
    mask[:,K-1,:observation_dim]=1 # TODO revisar
    
    assert mask.shape==(B,H,T)

    return mask



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