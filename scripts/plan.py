import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wandb
import torch
from datetime import datetime
from diffuser.sampling.policies import Policy_repaint_return_conditioned
from diffuser.utils.setup import load_experiment_params,set_seed
import diffuser.utils as utils
from diffuser.utils.rollouts import TrajectoryBuffer
from diffuser.utils.rendering import save_video,render_maze_2d

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

dataset="maze2d"
exp_name="gaussian_diff_returns_condition_H_128"
exp_name="gaussian_diff_returns_condition_H_128_1e-4_aug_ds"

args=load_experiment_params(f"logs/configs/{dataset}/{exp_name}/configs_diffusion.txt")

set_seed(args["seed"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## loading
current_dir=os.getcwd()

diffusion_loadpath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"diffusion", exp_name)

savepath=os.path.join(current_dir,args["logbase"], args["dataset_name"],"plans", exp_name)

os.makedirs(savepath,exist_ok=True)

wandb_log=False
return_chain=False
epoch="latest"

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    diffusion_loadpath,
    epoch=epoch, seed=args["seed"],
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
    resample_plan_every=125,
    ## sampling kwargs
    batch_size_sample=args["batch_size_sample"],
    horizon_sample = args["horizon_sample"],
    guidance_scale= 1.2, 
    temperature= 0.5,
    resample_every= 50,
    return_chain= return_chain
)

policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
 
if wandb_log:
    wandb.init(
        project='MT_inpainting_diffuser',
        name=exp_name,
        monitor_gym=True,
        save_code=True)
    
print(savepath, flush=True)

seed = 123 #int(datetime.now().timestamp()) # TODO maybe change this... 
print(f"Using seed:{seed}")



rollouts=TrajectoryBuffer(action_dim=dataset.action_dim)

# maze2d only
for episode in range(100):
    env = dataset.minari_dataset.recover_environment(render_mode="rgb_array_list")
    observation, info = env.reset(seed=seed+episode)

    rollouts.start_trajectory(first_observation=observation["observation"],first_info=info)
    total_reward = 0
    for t in range(args["max_episode_length"]):

        action, samples = policy(rollouts,provide_task=observation["desired_goal"])

        #next_waypoint = samples.observations[0,1,:]
       # actual_waypoint = samples.observations[0,0,:]

       # action = next_waypoint[:2] - actual_waypoint[:2] + (next_waypoint[2:] - actual_waypoint[2:])
       # action=action.cpu().numpy()
        ## execute action in environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward # clave que el max episode lenght sea el mismo que el con el que se recolecto el dataset, para el score, si no se tienen scores diferentes.
        
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | Infered_task: {samples.task} | Real_task: {observation["desired_goal"]}',
            #f'values: {samples.values} | scale: {args["scale"]}',
            flush=True,)

        rollouts.add_transition(observation["observation"], action, reward, terminated,total_reward,info) # maze2d

        if terminated or info["success"]==True or t==args["max_episode_length"]-1: # maze2d

            policy.resample_counter=0
            rollouts.end_trajectory()
            rollouts.save_trajectories(filepath=os.path.join(savepath,f'rollout_{seed}.pkl'))
            save_video(env,seed,savepath,suffix=episode,wandb_log=wandb_log)

            episode=rollouts.rollouts_to_numpy(index=-1)
            real_observations=episode.states
            render_maze_2d(env=dataset.env,observations=real_observations,goal_state=observation["desired_goal"],fig_name=f"maze_real_test_env_plan") # TODO task
#render_maze_2d(env=dataset.env,observations=observations.cpu().numpy()[0,:,:],goal_state=task.cpu().numpy()[0,0,:],fig_name=f"maze_sample_test_env{t}")
            import time
            time.sleep(5)
            break
        
# Close the environment
env.close()

## write results to json file at `args.savepa th`
logger.finish(t, total_reward, terminated, diffusion_experiment,seed,args["batch_size_sample"]) # TODO change to inside the terminated 



#IDEA LOOP
"""
1. crea env
2. loop sobre episodios (totrew=0),reset loggger
    3. loop sobre episodio, cuando termina (save video, logger, episode rollout, render map and render diffusion imagined map)
4. env close


Policy(task,rollouts)
infer task
infer action
return action

"""
