import os
import json


class Logger:

    def __init__(self, logpath):
        self.savepath = logpath

    def finish(self, t, total_reward, terminal, diffusion_experiment,seed,batch_size):
        json_path = os.path.join(self.savepath, f'results_{seed}.json')
        json_data = { 'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': diffusion_experiment.epoch,  "env_seed":seed,"batch_size_sample":batch_size}
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
        print(f'[ utils/logger ] Saved log to {json_path}')
