import numpy as np
import torch
from diffuser.utils.arrays import atleast_2d

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    """
    Base normalizer class

    Args:
        dataset (dict): dict with episodes, episode_id as keys and with dict of attributes as values
        normed_keys (list): Keys to normalize
    """
    def __init__(self,
                 dataset,
                 normed_keys
                 ):
        self.dataset=dataset
        self.make_params(normed_keys=normed_keys)


    def make_params(self,normed_keys=['observations', 'actions']):
        """
        Funcion to make the parameters that will be used in normalization.
        """

        keys_attribute_dict= {key: [] for key in normed_keys}

        ### make a list of episodes by key ### 
        for ep_id, dict in self.dataset.items():
            for key,attribute in dict.items():
                attribute=atleast_2d(attribute)

                if key in normed_keys:
                    keys_attribute_dict[key].append(attribute)
        
        ### get normed keys ###
        self.params_dict={}
        for key in normed_keys:
            print(f"Getting normalization params for {key} ...")
            concat_episodes_key=np.concatenate(keys_attribute_dict[key]) # all episodes concatenated per key... 
            mean=concat_episodes_key.mean(axis=0)
            std=concat_episodes_key.std(axis=0)  
            min=concat_episodes_key.min(axis=0)
            max=concat_episodes_key.max(axis=0)
            self.params_dict[key]={"mean":mean,"std":std,"min":min,"max":max}

          #  for key, value in self.params_dict[key].items(): # check dtypes
            #    assert value.dtype == np.float32, f"Value for key '{key}' is not of dtype float32"
    
    def __call__(self, x, key):
        return self.normalize(x, key)


    def normalize(self,unnormed_data, key):
        """
        Function to normalize unnormed data.
        
        Args: 
            unnormed_data (Torch.tensor or np.array): Data to norm
            key: Field of wich the data belongs to.

        Returns:
            normalized_data (Torch.tensor or np.array)
        """
        if isinstance(unnormed_data, np.ndarray):
            return self.normalize_numpy(unnormed_data, key)
        elif isinstance(unnormed_data, torch.Tensor):
            return self.normalize_torch(unnormed_data, key)
        else:
            raise TypeError("The object has to be a Torch tensor or a numpy array")


    def unnormalize(self,normed_data, key):
        """
        Function to unnormalize normed data.
        
        Args: 
            normed_data (Torch.tensor or np.array): Data to unnorm
            key: Field of wich the data belongs to.

        Returns:
            unnormalized_data (Torch.tensor or np.array)
        """
        if isinstance(normed_data, np.ndarray):
            return self.unnormalize_numpy(normed_data,key)
        elif isinstance(normed_data, torch.Tensor):
            return self.unnormalize_torch(normed_data,key)
        else:
            raise TypeError("The object has to be a Torch tensor or a numpy array")
        
    def normalize_numpy(self,unnormed_data, key):
        raise NotImplementedError


    def unnormalize_numpy(self,normed_data,key):
        raise NotImplementedError

    
    def normalize_torch(self,unnormed_data,key):
        raise NotImplementedError

    
    def unnormalize_torch(self,normed_data,key):
        raise NotImplementedError

    

class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, dataset, normed_keys):
        super().__init__(dataset, normed_keys)

    def normalize_numpy(self,unnormed_data, key):
        
        normed_data=(unnormed_data-self.params_dict[key]["mean"])/self.params_dict[key]["std"]

        return(normed_data)


    def unnormalize_numpy(self,normed_data,key):

        unnormed_data=normed_data*self.params_dict[key]["std"]+self.params_dict[key]["mean"]

        return(unnormed_data)
    
    
    def normalize_torch(self,unnormed_data,key):
        
        torch_mean=torch.from_numpy(self.params_dict[key]["mean"]).to(unnormed_data.device)
        torch_std=torch.from_numpy(self.params_dict[key]["std"]).to(unnormed_data.device)
        
        assert torch_mean.shape==torch_std.shape
        assert torch_mean.shape==unnormed_data[0,0,:].shape

        normed_data=(unnormed_data-torch_mean)/torch_std

        return(normed_data)
    
    
    def unnormalize_torch(self,normed_data,key):
        torch_mean=torch.from_numpy(self.params_dict[key]["mean"]).to(normed_data.device)
        torch_std=torch.from_numpy(self.params_dict[key]["std"]).to(normed_data.device)

        assert torch_mean.shape==torch_std.shape
        assert torch_mean.shape==normed_data[0,0,:].shape

        unnormed_data=normed_data*torch_std+torch_mean
        return(unnormed_data)


class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''
    def __init__(self,dataset,normed_keys):
        super().__init__(dataset,normed_keys)
    
    def normalize_numpy(self,unnormed_data, key): # TODO test... 
        
        ## [ 0, 1 ]
        normed_data=(unnormed_data-self.params_dict[key]["min"])/(self.params_dict[key]["max"] - self.params_dict[key]["min"])
        
        ## [ -1, 1 ]
        normed_data=2*normed_data-1

        return(normed_data)

    def unnormalize_numpy(self,normed_data,key,eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''

        if self.params_dict[key]["max"] > 1 + eps or self.params_dict[key]["min"] < -1 - eps:
            print(f'Warning: sample out of range | ({self.params_dict[key]["min"].min():.4f}, {self.params_dict[key]["max"].max():.4f})')
            normed_data = np.clip(normed_data, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        normed_data = (normed_data + 1) / 2.

        return normed_data * (self.params_dict[key]["max"] - self.params_dict[key]["min"]) + self.params_dict[key]["min"]
    

    def normalize_torch(self,unnormed_data,key):

        torch_min=torch.from_numpy(self.params_dict[key]["min"]).to(unnormed_data.device)
        torch_max=torch.from_numpy(self.params_dict[key]["max"]).to(unnormed_data.device)
        
        assert torch_min.shape==torch_max.shape
        assert torch_max.shape==unnormed_data[0,0,:].shape

        ## [ 0, 1 ]
        normed_data=(unnormed_data-torch_min)/(torch_max - torch_min)
        
        ## [ -1, 1 ]
        normed_data=2*normed_data-1

        return(normed_data)
    
    def unnormalize_torch(self,normed_data,key,eps=1e-4):

        torch_min=torch.from_numpy(self.params_dict[key]["min"]).to(normed_data.device)
        torch_max=torch.from_numpy(self.params_dict[key]["max"]).to(normed_data.device)
        
        assert torch_min.shape==torch_max.shape
        assert torch_max.shape==normed_data[0,0,:].shape

        if (torch_max > 1 + eps).any() or (torch_min < -1 - eps).any(): # ver si se puede hacer comparacion... 
            print(f'Warning: sample out of range | ({torch_min.min():.4f}, {torch_max.max():.4f})')
            normed_data = torch.clip(normed_data, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        normed_data = (normed_data + 1) / 2.

        return normed_data * (torch_max- torch_min) + torch_min

    
class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        functions like LimitsNormalizer, but can handle data for which a dimension is constant
    '''

    def __init__(self,key, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.params_dict[key]["min"])): # ojo que este lop hace para todas las dimensiones  de una... cambiar
            if self.params_dict[key]["min"][i] == self.params_dict[key]["max"][i]:
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.params_dict[key]["max"][i]}'''
                )
                self.params_dict[key]["min"] -= eps
                self.params_dict[key]["max"] += eps