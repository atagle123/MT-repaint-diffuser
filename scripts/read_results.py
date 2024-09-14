import os
import numpy as np
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_results(folder_path, key):
    results_list = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a JSON file
        if filename.endswith('.json'):
            results_list.append(load_result(file_path,key))

    return results_list

def get_score(folder_path,min_reward,max_reward):
	'''
		paths : path to directory containing experiment trials
	'''
	returns=load_results(folder_path,"return")
	n_samples=len(returns)
	returns=np.array(returns)
	scores=(returns-min_reward)/(max_reward-min_reward)
	scores*=100
	if len(scores) > 0:
		mean = np.mean(scores)
		min=np.min(scores)
		max=np.max(scores)
	else:
		mean = np.nan
		min=np.nan
		max=np.nan
	if len(scores) > 1:
		err = np.std(scores) / np.sqrt(len(scores))
	else:
		err = 0
	return mean, err, scores,min, max, n_samples

def get_min_max_reward(dataset_name):
	""" Function that gets the min max reward, from the infos.py data

	"""
	from infos import REF_MAX_SCORE,REF_MIN_SCORE
	max_reward=REF_MAX_SCORE[dataset_name]
	min_reward=REF_MIN_SCORE[dataset_name]
	return(min_reward,max_reward)

def load_result(file_path,key):
	'''
		path : path to experiment directory; expects `rollout.json` to be in directory
	'''
	with open(file_path, 'r') as file:
		try:
			data = json.load(file)
			# Extract values associated with the specified key
			if key in data:
				result=data[key]
		except json.JSONDecodeError as e:
			print(f"Error decoding JSON in {file_path}: {str(e)}")
		except KeyError as e:
			print(f"KeyError: '{key}' not found in {file_path}")
	return(result)


#######################
######## setup ########
#######################


if __name__ == '__main__':

	current_dir=os.getcwd()
	exps_dirs=os.path.join(current_dir,"logs/pretrained")
	for dataset_name in os.listdir(exps_dirs):
		file_path= os.path.join(exps_dirs, dataset_name,"plans")

		if os.path.exists(file_path):
			for exp_name in os.listdir(file_path):
				exp_path=os.path.join(file_path ,exp_name)
				min_reward,max_reward=get_min_max_reward(dataset_name)
				mean,error,_, min, max, n_samples =get_score(folder_path=exp_path,min_reward=min_reward,max_reward=max_reward)

				print(f"dataset: {dataset_name} exp_name: {exp_name} mean:{mean}",f"error:{error}", f"min: {min} max: {max}", f"n samples: {n_samples}")
