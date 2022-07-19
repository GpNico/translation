

import argparse
import os
import yaml
from analysis.multiple_runner import MultipleRunner

from src.train import BackTranslation
from analysis.analyser import Analyser2D
from analysis.multiple_runner import MultipleRunner
from src.tune import HyperParametersTuner
from tools.helpers import get_configs_name

if __name__ == '__main__':
    
    
    # To be changed
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='ContinuousSquares',
                        dest='dataset',
                        help='Name of the dataset. Choices : ContinuousSquares, GaussianMixture.')
    parser.add_argument("--train", 
                        help="If true it runs a training.",
                        action="store_true")
    parser.add_argument("--plot", 
                        help="If true it plot the analysis.",
                        action="store_true")
    parser.add_argument("-a",
                        "--analyse", 
                        help="If true it computes an analysis.",
                        action="store_true")
    parser.add_argument("-m",
                        "--multiple_exp", 
                        help="If true conducts multiple experiment, compute stats \
                              and save it.",
                        action="store_true")
    parser.add_argument("--tune", 
                        help="Run hyperparameters search.",
                        action="store_true")
    args = parser.parse_args()
    
    assert not(args.multiple_exp) or (args.train == False and args.analyse == False)

    # Configs
    selector_configs_file = os.path.join('configs',args.dataset,'vanilla.yml')

    with open(selector_configs_file) as file:
        selector_configs = yaml.safe_load(file)

    best_configs_file = os.path.join('configs',
                                     args.dataset,
                                     '{}.yml'.format(get_configs_name(selector_configs))
                                     )

    try:
        with open(best_configs_file) as file:
            configs = yaml.safe_load(file)
    except:
        print("This configs parameters were not optimized!")
        configs = selector_configs

    # Options
    if args.train:
        # Training
        bt = BackTranslation(configs)
        bt.train(silent = False,
                 analyse = args.analyse)
    
    if args.analyse:
        if configs['name'] in ['DiscreteSquares', 
                               'ContinuousSquares',
                               'GaussianMixture']:
            analysis_2d = Analyser2D(configs)

            analysis_2d.L_and_M_stats(n_batches = 8,
                                      plot = args.plot)

    if args.multiple_exp:
        runner = MultipleRunner(configs,
                                n_exp = 30)
        runner.stats_wrapper()


    # To put in its own file afterwards
    if args.tune:
        tuner = HyperParametersTuner(configs)
        tuner.tune(num_samples = 20)


        

        
        
