import h5py
import os
import json
from tqdm import tqdm
import numpy as np


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(self, h5path=None):
    #if h5path is None:
        #  if self._dataset_url is None:
        #      raise ValueError("Offline env not configured with a dataset URL.")
        #  h5path = download_dataset_from_url(self.dataset_url)

    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if self.observation_space.shape is not None:
        assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
            'Observation shape does not match env: %s vs %s' % (
                str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
    assert data_dict['actions'].shape[1:] == self.action_space.shape, \
        'Action shape does not match env: %s vs %s' % (
            str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    return data_dict


def step_tuple_to_traj_tuple(obs, act, rew, next_obs, term, trunc):
    dones = np.logical_or(term, trunc)[:-1]  # last one should not be used for split to avoid empty chunk
    dones_ind = np.where(dones)[0] + 1
    obs, act, rew, next_obs, term, trunc = \
        map(lambda x: np.split(x, dones_ind), [obs, act, rew, next_obs, term, trunc])

    obs_new = [np.concatenate([_obs, _next_obs[-1].reshape(1, -1)])
               for _obs, _next_obs in zip(obs, next_obs)]
    buffer = []
    keys = ['observations', 'actions', 'rewards', 'terminations', 'truncations']
    for _traj_dt in zip(obs_new, act, rew, term, trunc):
        _buff_i = dict(zip(keys, _traj_dt))
        buffer.append(_buff_i)
    return buffer

def make_traj_based_buffer(d4rl_env_name):
    env = gym.make(d4rl_env_name)
    dt = env.get_dataset()
    obs = dt['observations']
    next_obs = dt['next_observations']
    rewards = dt['rewards']
    actions = dt['actions']
    terminations = dt['terminals']
    truncations = dt['timeouts']

    buffer = step_tuple_to_traj_tuple(obs, actions, rewards, next_obs, terminations, truncations)

    return buffer, env


if __name__=="__main__":
    pass