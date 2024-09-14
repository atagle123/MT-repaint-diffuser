import concurrent.futures
import subprocess
import os
import time
import random


def run_plan_guide(conda_env,num_batches,script):
    current_dir=os.getcwd()
    user_dir=os.path.expanduser("~")

    time.sleep(2) # importante, es para que tengan semillas distintas al principio

    script_execution=f"{user_dir}/miniconda3/envs/{conda_env}/bin/python {current_dir}/{script}"

    scripts_execution=repeat_string_with_separator(script_execution,num_batches,sep=" ; ")

    command=f'gnome-terminal -- bash -c "{scripts_execution}"'

    process=subprocess.Popen(command,shell=True)
    process.wait()
    return(process)

def repeat_string_with_separator(s, n,sep=" ; "):
    # Create a list containing the repeated string
    repeated_strings = [s] * n
    
    # Join the list elements with " ; "
    result = sep.join(repeated_strings)
    
    return result


def main():
    conda_env="thesis"
    num_workers=1 # importante para que no tenga la  misma seed, solo se debe abrir un terminal a la vez 
    num_tasks_per_batch=6
    num_batches=5

    script="scripts/plan.py"

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_plan_guide, conda_env,num_batches,script) for _ in range(num_tasks_per_batch)]
    
        concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    

if __name__=="__main__":
    main()