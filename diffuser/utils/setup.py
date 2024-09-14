import os
import importlib
import random
import numpy as np
import torch

from .serialization import mkdir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True


# Function to load experiment parameters from text file
def load_experiment_params(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Evaluate value to handle different data types
                if value.lower() == 'none':
                    params[key] = None
                elif value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                else:
                    try:
                        params[key] = eval(value)
                    except (NameError, SyntaxError):
                        params[key] = value  # Fallback to string if eval fails

    return params






