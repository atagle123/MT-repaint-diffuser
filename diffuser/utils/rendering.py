import os
import numpy as np
import einops
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gymnasium
import mujoco as mjc
import warnings
import pdb
import wandb
import imageio.v2 as iio
from .arrays import to_np


#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)



#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)

def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))



def render_maze_2d(env,observations,goal_state,fig_name="maze2d_plot",title="Maze2d map"):
    """    MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12)
    }"""

    maze_size_scaling=1

    back=env.spec.kwargs["maze_map"]
    plt.grid(True)
    back=np.rot90(np.array(back), k=2).T
    map_length = len(back)
    map_width = len(back[0])
    x_map_center = map_width / 2 * maze_size_scaling
    y_map_center = map_length / 2 * maze_size_scaling

    plt.clf()
    fig = plt.gcf()

    extent = [
        x_map_center - map_width ,  # Left
        x_map_center ,  # Right
        y_map_center - map_length , # Bottom
        y_map_center   # Top
    ]

    plt.imshow(back,extent=extent,
        cmap=plt.cm.binary, vmin=0, vmax=1)

    path_length = len(observations)
    colors = plt.cm.jet(np.linspace(0,1,path_length))
    plt.plot(goal_state[1], goal_state[0],marker='*', markersize=10, color='r', zorder=12)
    plt.plot(observations[0,1], observations[0,0],marker='*', markersize=10, color='g', zorder=12)
    plt.plot(observations[:,1], observations[:,0], c='black', zorder=11)
    plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=10)
    plt.title(title)
    plt.savefig(f'{fig_name}.png', dpi=300) #   TODO : save path


def save_video(env,seed,savepath,suffix,wandb_log=False):
    """
    Asumes that env is in render mode
    """
    filename = f'rollout_video_{seed}_{suffix}.mp4'
    filepath=os.path.join(savepath,filename)
    writer = iio.get_writer(filepath, fps=env.metadata["render_fps"])

    frames=env.render()
    for frame in frames:
        frame = (frame * -255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()

    if wandb_log: wandb.log({"video": wandb.Video(filepath)})
    pass