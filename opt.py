import argparse

import nni
from nni.experiment import Experiment
parser = argparse.ArgumentParser(description='opt')
parser.add_argument("--data_choice","-d", default="DBP15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K"],
                    help="Experiment path")
parser.add_argument("--data_rate", type=float, default=0.3)
parser.add_argument("--data_split", type=str, default='zh_en')
parser.add_argument("--gpu", type=int, default=0,
                    help="Which GPU to use?")
parser.add_argument("--experiment_name", "-e", type=str, default="default",
                    help="A folder with this name would be created to dump saved models and log files")
parser.add_argument("--port", "-p", type=int, default=8081)


params = parser.parse_args()

search_space = {
        "n_layer": {"_type": "choice", "_value": [5]},
        "n_batch": {"_type": "choice", "_value": [4,5,6]},
        "lr": {"_type": "loguniform", "_value": [0.0001, 0.01]},
        "lamb": {"_type": "uniform", "_value": [0.00001, 0.005]},
        "dropout": {"_type": "uniform", "_value": [0, 0.5]},
        "act": {"_type": "choice", "_value": ["relu", "idd", "tanh"]},
        "hidden_dim": {"_type": "choice", "_value": [64,128,200]},
        "decay_rate": {"_type": "uniform", "_value": [0.99, 1]},
        "attn_dim": {"_type": "choice", "_value": [5,10,20,40]},
    }


experiment = Experiment('local')
cmd = f'python ./train.py --nni 1 --data_split {params.data_split}'

experiment.config.trial_command = cmd
experiment.config.trial_code_directory = '.'

# experiment.config.experiment_working_directory = './experiments'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 2
experiment.config.max_trial_duration = '240h'
# experiment.config.training_service.gpu_indices = [0,1,2,3]
experiment.config.training_service.gpu_indices = [0,1]
experiment.config.trial_gpu_number = 1
experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.max_trial_number_per_gpu = 1
experiment.run(params.port)
