import gymnasium as gym
import minari
import numpy as np
import h5py
from tqdm import tqdm

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

def get_tuple_from_minari_dataset(dataset_name):
    dt = minari.load_dataset(dataset_name)
    observations, actions, rewards, next_observations, terminations, truncations = \
        [], [], [], [], [], []
    traj_length = []
    for _ep in dt:
        observations.append(_ep.observations[:-1])
        actions.append(_ep.actions)
        rewards.append(_ep.rewards)
        next_observations.append(_ep.observations[1:])
        terminations.append(_ep.terminations)
        truncations.append(_ep.truncations)
        traj_length.append(len(_ep.rewards))
        assert (_ep.truncations[-1] or _ep.terminations[-1])
    observations, actions, rewards, next_observations, terminations, truncations = \
        map(np.concatenate, [observations, actions, rewards, next_observations, terminations, truncations])
    traj_length = np.array(traj_length)
    return observations, actions, rewards, next_observations, terminations, truncations, traj_length


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


def create_standard_d4rl():

    mujoco_envs = ['Hopper', 'HalfCheetah', 'Ant', 'Walker2d']
    quality_lists = ['expert', 'medium', 'random', 'medium-expert',"medium-replay"]

    for _env_prefix in mujoco_envs:
        for _quality in quality_lists:
            env_name = f'{_env_prefix.lower()}-{_quality}-v2'

            buffer, env = make_traj_based_buffer(env_name)
            if not (buffer[-1]["terminations"][-1] or buffer[-1]["truncations"][-1]):
                buffer[-1]["truncations"][-1] = True

            gymnasium_env = gym.make(f'{_env_prefix}-v2')
            dataset = minari.create_dataset_from_buffers(
                dataset_id=env_name,
                env=gymnasium_env,
                buffer=buffer,
                algorithm_name='SAC',
                author='Zhiyuan',
                # minari_version=f"{minari.__version__}",
                author_email='levi.huzhiyuan@gmail.com',
                code_permalink='TODO',
                ref_min_score=env.ref_min_score,
                ref_max_score=env.ref_max_score,
            )
            print('dataset created')
    return


def validate_standard_d4rl():
    mujoco_envs = ['Hopper', 'HalfCheetah', 'Ant', 'Walker2d']
    quality_lists = ['expert', 'medium', 'random', 'medium-expert']

    for _env_prefix in mujoco_envs:
        for _quality in quality_lists:
            env_name = f'{_env_prefix.lower()}-{_quality}-v2'

            minari_tuple = get_tuple_from_minari_dataset(env_name)
            m_obs, m_act, m_rew, m_next_obs, m_term, m_trunc, m_traj_len = minari_tuple

            d4rl_data = gym.make(f'{_env_prefix.lower()}-{_quality}-v2').get_dataset()
            assert np.all(m_act == d4rl_data["actions"])
            assert np.all(m_obs == d4rl_data["observations"])
            assert np.all(m_next_obs == d4rl_data["next_observations"])
            assert np.all(m_rew == d4rl_data["rewards"])
            assert np.all(m_term == d4rl_data["terminals"])
            assert np.all(m_trunc[:-1] == d4rl_data["timeouts"][:-1])
            assert m_trunc[-1]

            d4rl_dones = np.logical_or(d4rl_data["terminals"], d4rl_data["timeouts"])[:-1]
            # last one will always be added

            d4rl_dones = np.where(d4rl_dones)[0]
            num_d4rl = len(d4rl_data["rewards"])
            d4rl_dones = np.concatenate([[-1], d4rl_dones, [num_d4rl - 1]])
            d4rl_traj_length = d4rl_dones[1:] - d4rl_dones[:-1]
            assert np.all(d4rl_traj_length == m_traj_len)
            assert np.sum(m_traj_len) == len(m_rew)
            print('validation passed')
    return




if __name__=="__main__":
    create_standard_d4rl()
    validate_standard_d4rl()