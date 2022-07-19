import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import numpy as np
import yaml

from src.train import BackTranslation
from tools.helpers import get_configs_name

class HyperParametersTuner:
    """
        Class that compute optimal hyper-parameters
        using raytune lib.
    """

    def __init__(self, configs):
        # deactivate warnings
        #ray.init(log_to_driver=False)

        configs['training']['lr'] = tune.loguniform(1e-5, 1e-2)
        configs['training']["batch_size"] = tune.choice([16, 32, 64])

        if configs['models']['encoder']['model'] == 'mlp':
            configs['models']['encoder']["mlp"]["dim_hidden"] = tune.choice([4, 8, 16])
        if configs['models']['decoder']['model'] == 'mlp':
            configs['models']['decoder']["mlp"]["dim_hidden"] = tune.choice([4, 8, 16])

        if configs['training']['adversial']:
            configs['training']['lr adversial'] = tune.loguniform(1e-5, 1e-1)
            configs['training']['adversial_patience'] = tune.choice([1, 4, 16, 64, 128])

            if configs['models']['discriminator']['model'] == 'mlp':
              configs['models']['discriminator']["mlp"]["dim_hidden"] = tune.choice([4, 8, 16]) 

        self.configs = configs


        # Scheduler
        self.scheduler = ASHAScheduler(
                                max_t=configs['training']['n_epochs'],
                                grace_period=3,
                                reduction_factor=2
                                )

    def _train_bt(self, configs):

        N_models = 10
        
        bts = [BackTranslation(configs, 
                               ray_tune = True) for _ in range(N_models)]
        
        for _ in range(configs['training']['n_epochs']):
            total_valid_loss = []
            for k in range(N_models):
                _, valid_metrics = bts[k]._train_one_epoch(False)

                total_valid_loss.append(valid_metrics['total'])
            
            print("Losses : ", total_valid_loss)

            tune.report(loss = np.array(total_valid_loss).mean())
        

    def tune(self, num_samples: int = 10):
        """
            Run the tuning.
        """
        result = tune.run(
            tune.with_parameters(self._train_bt),
            resources_per_trial={"cpu": 2, "gpu": 1},
            config = self.configs,
            metric = "loss",
            mode = "min",
            num_samples = num_samples,
            scheduler = self.scheduler
        )

        best_trial = result.get_best_trial("loss", 
                                           "min", 
                                           "last")

        self.save_configs(best_trial.config)


    def save_configs(self, configs):
        cwd = os.getcwd()
        configs_file = os.path.join(
                        cwd, 
                        "configs\\{}\\{}.yml".format(
                                            configs['name'],
                                            get_configs_name(configs)
                                            )
                        )
        with open(configs_file, 'w') as file:
            yaml.dump(configs, file)

        print("Configs saved at {}".format(
                            configs_file
                            ))