import numpy as np

def find_key_in_data(data, key_to_find):
    """
    Function to find the value of a key in a annidated dict, it assumes that the value corresponds to a np.array
    """
    def recursive_search(item):
        # If item is a dictionary
        if isinstance(item, dict):
            if key_to_find in item:
                result = item[key_to_find]
                if isinstance(result, np.ndarray):
                    return result
                # Continue searching in values if the result is not found
            for value in item.values():
                result = recursive_search(value)
                if result is not None:
                    return result
        
        # If item is a list
        elif isinstance(item, list):
            for element in item:
                result = recursive_search(element)
                if result is not None:
                    return result
        
        # If item is a NumPy array (though arrays don't have keys, skip them)
        elif isinstance(item, np.ndarray):
            return None
        
        # If item has attributes
        elif hasattr(item, '__dict__'):
            if key_to_find in item.__dict__:
                result = getattr(item, key_to_find)
                if isinstance(result, np.ndarray):
                    return result
            for value in item.__dict__.values():
                result = recursive_search(value)
                if result is not None:
                    return result
        
        # If item has a method named `key_to_find`
        elif hasattr(item, key_to_find):
            result = getattr(item, key_to_find)
            if isinstance(result, np.ndarray):
                return result
        
        return None
    
    return recursive_search(data)



"""def make_returns(self): # TODO idea gamma deberia ser el mismo con el que se testea ... 
    print("Making returns... ")

    discount_array=self.discount ** np.arange(self.max_path_length) # (H)
    discount_array=atleast_2d(discount_array) # (H,1)
    for ep_id, dict in self.episodes.items():
        rtg=[]
        rewards=dict["rewards"]
        for i in range(len(rewards)):
            future_rewards=rewards[i:]
            horizon=len(future_rewards)
            norm_rtg=self.calculate_norm_rtg(future_rewards, horizon=horizon, discount=self.discount, discount_array=discount_array[:horizon]) # es enecesario normalizar rtg 
            rtg.append(np.exp(norm_rtg)) 
        
        returns_array=np.array(rtg,dtype=np.float32)
        assert returns_array.shape[0]==rewards.shape[0]

        self.episodes[ep_id]["returns"]=atleast_2d(returns_array)

    self.normed_keys.append("returns")"""