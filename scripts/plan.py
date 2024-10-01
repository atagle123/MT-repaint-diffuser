import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from diffuser.sampling.policies import Policy_repaint_return_conditioned
from diffuser.utils.setup import load_experiment_params,set_seed
import diffuser.utils as utils
import wandb
import imageio.v2 as iio
import numpy as np
import torch
from datetime import datetime
from diffuser.utils.rollouts import TrajectoryBuffer
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

dataset="maze2d"
exp_name="gaussian_diff_returns_condition"

args=load_experiment_params(f"logs/configs/{dataset}/{exp_name}/configs_diffusion.txt")

set_seed(args["seed"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## loading
current_dir=os.getcwd()

diffusion_loadpath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"diffusion", exp_name)

savepath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"plans", exp_name)

os.makedirs(savepath,exist_ok=True)

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    diffusion_loadpath,
    epoch="latest", seed=args["seed"],
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
#dataset.inference_mode()


logger_config = utils.Config(
    utils.Logger,
    logpath=savepath
)

logger = logger_config()

policy_config = utils.Config(
    Policy_repaint_return_conditioned,
    diffusion_model=diffusion,
    dataset=dataset,
    gamma=args["gamma"],
    keys_order=("observations","actions","rewards","task"), # TODO maybe this should be an attribute of dataset...
    ## sampling kwargs
    batch_size_sample=args["batch_size_sample"],
    horizon_sample = args["horizon_sample"],
    return_chain=args["return_chain"]
)

policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
 
wandb_log=args["wandb_log"]

if wandb_log:
    wandb.init(
        project='MT_inpainting_diffuser',
        name=exp_name,
        monitor_gym=True,
        save_code=True)
    
print(savepath, flush=True)

def save_video(env,seed,savepath,suffix,wandb_log=False):
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

seed = 123 #int(datetime.now().timestamp()) # TODO maybe change this... 
print(f"Using seed:{seed}")

env = dataset.minari_dataset.recover_environment(render_mode="rgb_array_list")
observation, info = env.reset(seed=seed)
# maze2d only
for episode in range(100):

    rollouts=TrajectoryBuffer(observation["observation"],info,action_dim=dataset.action_dim)
    total_reward = 0
    for t in range(args["max_episode_length"]):
    # print(observation,"observation")
        action, samples = policy(rollouts,provide_task=observation["desired_goal"])
    # print(action)
    # print("infered_task",samples.task)
        ## execute action in environment
        observation, reward, terminated, truncated, info = env.step(action)
        ## print reward and score
        total_reward += reward
        # clave que el max episode lenght sea el mismo que el con el que se recolecto el dataset, para el score, si no se tienen scores diferentes.
        
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | I_task: {samples.task:.2f} | R_task: {observation["desired_goal"]}',
            #f'values: {samples.values} | scale: {args["scale"]}',
            flush=True,)

        rollouts.add_transition(observation["observation"], action, reward, terminated,total_reward,info) # maze2d

        if terminated or info["success"]==True or t==args["max_episode_length"]-1: # maze2d
            rollouts.end_trajectory()
            rollouts.save_trajectories(filepath=os.path.join(savepath,f'rollout_{seed}.pkl'))
            save_video(env,seed,savepath,suffix=episode,wandb_log=wandb_log)
            break
        
# Close the environment
env.close()

## write results to json file at `args.savepath`
logger.finish(t, total_reward, terminated, diffusion_experiment,seed,args["batch_size_sample"])
