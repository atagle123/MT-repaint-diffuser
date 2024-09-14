import collections
import numpy as np
import torch
import pdb

DTYPE = torch.float
DEVICE = 'cuda:0'

#-----------------------------------------------------------------------------#
#------------------------------ numpy <--> torch -----------------------------#
#-----------------------------------------------------------------------------#

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	elif type(x) is dict:
		return {k: to_np(v) for k, v in x.items()}
	return x

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
	return torch.tensor(x, dtype=dtype, device=device)

def to_device(x, device=DEVICE,dtype=torch.float):
	if torch.is_tensor(x):
		x=x.float()
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v.float(), device) for k, v in x.items()}
	else:
		raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')

def batchify(batch):
	'''
		convert a single dataset item to a batch suitable for passing to a model by
			1) converting np arrays to torch tensors and
			2) and ensuring that everything has a batch dimension
	'''
	fn = lambda x: to_torch(x[None])

	batched_vals = []
	for field in batch._fields:
		val = getattr(batch, field)
		val = apply_dict(fn, val) if type(val) is dict else fn(val)
		batched_vals.append(val)
	return type(batch)(*batched_vals)

def apply_dict(fn, d, *args, **kwargs):
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}

def normalize(x):
	"""
		scales `x` to [0, 1]
	"""
	x = x - x.min()
	x = x / x.max()
	return x

def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1,2,0))
    return (array * 255).astype(np.uint8)

def set_device(device):
	DEVICE = device
	if 'cuda' in device:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)

def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

#-----------------------------------------------------------------------------#
#----------------------------- parameter counting ----------------------------#
#-----------------------------------------------------------------------------#

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def report_parameters(model, topk=10):
	counts = {k: p.numel() for k, p in model.named_parameters()}
	n_parameters = sum(counts.values())
	print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

	modules = dict(model.named_modules())
	sorted_keys = sorted(counts, key=lambda x: -counts[x])
	max_length = max([len(k) for k in sorted_keys])
	for i in range(topk):
		key = sorted_keys[i]
		count = counts[key]
		module = param_to_module(key)
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


def pad(array,max_len, pad_value=0):
	"""
	Pad the input array to a fixed length with a specified padding value.

	Parameters:
	- array: Input array (list or numpy array).
	- max_len: Desired fixed length of the padded array.
	- pad_value: Value used for padding (default is 0).

	Returns:
	- Padded array of length fixed_length.
	"""

	current_length = len(array)

	if current_length >= max_len:
		return array[:max_len]  # Return trimmed array if longer or equal
	
	pad_shape = (max_len - current_length,) + array.shape[1:]

	# Create padding array filled with pad_value
	padding = np.full(pad_shape, pad_value, dtype=array.dtype)
	
	# Concatenate original array and padding
	padded_array = np.concatenate((array, padding), axis=0)
	
	return padded_array

def pad_min(array,min_len, pad_value=0):
	"""
	Pad_min the input array to a fixed length with a specified padding value.

	Parameters:
	- array: Input array (list or numpy array).
	- max_len: Desired fixed length of the padded array.
	- pad_value: Value used for padding (default is 0).

	Returns:
	- Padded array of length fixed_length.
	"""

	current_length = len(array)

	if current_length >= min_len:
		return array  # Return trimmed array if longer or equal
	
	pad_shape = (min_len - current_length,) + array.shape[1:]

	# Create padding array filled with pad_value
	padding = np.full(pad_shape, pad_value, dtype=array.dtype)
	
	# Concatenate original array and padding
	padded_array = np.concatenate((array, padding), axis=0)
	
	return padded_array

### torch functions ###

def relu(input_tensor,max):
    # Create a new tensor with the same shape as input_tensor
    output_tensor = torch.zeros_like(input_tensor)
    # Apply ReLU element-wise
    output_tensor[input_tensor >= max] = input_tensor[input_tensor >= max]
    return output_tensor

def symetric_relu(input_tensor,min):
	"""
	\___
	"""
	# Create a new tensor with the same shape as input_tensor
	output_tensor = torch.zeros_like(input_tensor)
	# Apply ReLU element-wise
	output_tensor[input_tensor <= min] = -input_tensor[input_tensor <= min]
	return output_tensor

def pozo_relu(input_tensor,min,max):
	"""
	\___/
	
	"""
	output_tensor = torch.zeros_like(input_tensor)
	output_tensor[input_tensor <= min] = -input_tensor[input_tensor <= min]
	output_tensor[input_tensor >= max] = input_tensor[input_tensor >= max]

	return output_tensor


def symetric_relu_grad(input_tensor,min):
	"""
	\___
	"""
	# Create a new tensor with the same shape as input_tensor
	output_tensor = torch.zeros_like(input_tensor)
	# Apply ReLU element-wise
	output_tensor[input_tensor <= min] = -1
	return output_tensor

def pozo_relu_grad(input_tensor,min,max):
	"""
	\___/
	
	"""
	output_tensor = torch.zeros_like(input_tensor)
	output_tensor[input_tensor <= min] = -1
	output_tensor[input_tensor >= max] = 1

	return output_tensor

