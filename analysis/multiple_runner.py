"""
    blabla
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pickle

from src.train import BackTranslation
from analysis.analyser import Analyser2D

class MultipleRunner:
    """
        Wrapper that run an experiment multiple times,
        compute associated stats and then save it.
        /!\ In 2D /!\
    """
    def __init__(self, configs: dict,
                       n_exp: int = 3) -> None:
        self.configs = configs

        # Stats
        self.n_exp = n_exp
        self.stats = {}


    def stats_wrapper(self):
        """
            Compute and save desired statistics.
        """
        # Name of the config
        csv_file = "analysis\\reports\\{}.csv".format(
                                        self.configs['name']
                                        )
        pickle_file = "analysis\\reports\\{}".format(
                                        self.configs['name']
                                        )
        # Load it or create it
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, sep = ";")
        else:
            df = pd.DataFrame([], columns = self._columns_names())

        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                raw_dict = pickle.load(f)
        else:
            raw_dict = {}

        # Get columns
        columns = list(df.columns)
        
        # Get row
        config_list = self._config_filler()
        stats_list, stats_raw = self._stats_filler()

        config_key = self._get_key_from_config_list(config_list)
        raw_dict[config_key] = stats_raw

        line_idx = self._check_config_line(df, 
                                           config_list)

        full_row = config_list + stats_list

        if line_idx == -1:
            df = df.append({columns[k]: full_row[k] 
                                for k in range(len(columns))},
                            ignore_index = True)
        else:
            print("Warning: there are already computed stats with these configs. Erasing them...")
            df.iloc[line_idx] = full_row
        
        df.to_csv(csv_file, sep = ';', index = False)

        with open(pickle_file, 'wb') as f:
                pickle.dump(raw_dict, f)


    def _stats_filler(self) -> list:
        """
            Get the stats and return it.
        """

        stats = [{} for _ in range(self.configs['training']['n_epochs'] + 1)]

        for n in range(self.n_exp):
            print("\n ###################### EXP {}\{} #######################\n".format(n+1,
                                                                                         self.n_exp))
            bt = BackTranslation(self.configs)

            # Initial Data
            if self.configs['name'] in ['DiscreteSquares', 
                                        'ContinuousSquares',
                                        'GaussianMixture']:
                    analysis_2d = Analyser2D(self.configs,
                                             load_weights = False,
                                             L1_test_loader = bt.L1_test_loader,
                                             L2_test_loader = bt.L2_test_loader,
                                             model = bt.model,
                                             M_discriminator = bt.M_discriminator)

                    analysis_2d.L_and_M_stats(n_batches = len(analysis_2d.L1_test_loader),
                                              plot = False)
            else:
                raise Exception('config {} analyser has not been implemented.'.format(self.configs['name']))

            stats[0] = self._fus_dict(stats[0], analysis_2d.stats)

            for epoch in range(self.configs['training']['n_epochs']):

                _, valid_metrics = bt._train_one_epoch(False)

                if not(False):
                    print("### Epoch {} ###".format(epoch))
                    print('\t BT Valid : L1 Loss {} | L2 Loss {}'.format(
                                                                    valid_metrics['BT L1'],
                                                                    valid_metrics['BT L2']
                                                                    ))
                    if self.configs['training']['denoising']:
                        print('\t DAE Valid : L1 Loss {} | L2 Loss {}'.format(
                                                                    valid_metrics['DAE L1'],
                                                                    valid_metrics['DAE L2']
                                                                    ))
                    if self.configs['training']['adversial']:
                        print('\t Discriminator : Loss {} | Acc. {}'.format(
                                                                    valid_metrics['D Loss'],
                                                                    valid_metrics['D Acc']
                                                                    ))

                    print('\t L1 trans. err. {} | L2 trans. err.  {}'.format(
                                                                    valid_metrics['Trans L1'],
                                                                    valid_metrics['Trans L2']
                                                                    ))

                if self.configs['name'] in ['DiscreteSquares', 
                                            'ContinuousSquares',
                                            'GaussianMixture']:
                    analysis_2d.L_and_M_stats(n_batches = len(analysis_2d.L1_test_loader),
                                              plot = False)
                else:
                    raise Exception('config {} analyser has not been implemented.'.format(self.configs['name']))

                stats[epoch+1] = self._fus_dict(stats[epoch+1], analysis_2d.stats)

        stats_list_raw = []
        for epoch in range(len(stats)):
            stats_list_raw.append(np.stack((stats[epoch]['Ms overlap score'],
                                            stats[epoch]['untrained Ms overlap score'],
                                            stats[epoch]['L1s overlap score'],
                                            stats[epoch]['L2s overlap score'],
                                            stats[epoch]['trans err L1'],
                                            stats[epoch]['trans err L2'])))

        return stats_list_raw[-1].mean(axis = 1).tolist(), np.stack(stats_list_raw)

    def _config_filler(self) -> list[float]:
        """
            Look at the configs file and returns a list
            that specifies with ones and zeros wheter the
            configs has this particular attributes.
            Also specifies the language attributes.
        """

        if self.configs['name'] == 'ContinuousSquares':
            config_list = [self.configs['dataset']['size'],
                        self.configs['training']['prop_gold'],
                        float(self.configs['training']['denoising']),
                        float(self.configs['training']['adversial']),
                        float(self.configs['training']['share_enc']),
                        float(self.configs['training']['share_dec']),
                        self.configs['models']['encoder']['model'],
                        self.configs['models']['encoder']['batch_norm'],
                        self.configs['models']['decoder']['model'],
                        self.configs['models']['discriminator']['model']\
                        if self.configs['training']['adversial'] else "None",
                        self.configs['dataset']['L1']['probs']['type'],
                        self.configs['dataset']['L2']['probs']['type']]
        elif self.configs['name'] == 'GaussianMixture':
            config_list = [self.configs['dataset']['size'],
                        self.configs['training']['prop_gold'],
                        float(self.configs['training']['denoising']),
                        float(self.configs['training']['adversial']),
                        float(self.configs['training']['share_enc']),
                        float(self.configs['training']['share_dec']),
                        self.configs['models']['encoder']['model'],
                        self.configs['models']['encoder']['batch_norm'],
                        self.configs['models']['decoder']['model'],
                        self.configs['models']['discriminator']['model']\
                        if self.configs['training']['adversial'] else "None",
                        'gaussian_mixture',
                        'gaussian_mixture']

        return config_list

    def _columns_names(self) -> list[str]:
        return ['dataset size',
                'prop gold',
                'denoising',
                'adversial',
                'share enc',
                'share dec', 
                'encoder model',
                'encoder BN',
                'decoder model',
                'discriminator model',
                'L1 probs type',
                'L2 probs type', # Last config column
                'Ms overlap score', # From here Stats
                'untrained Ms overlap score',
                'L1s overlap score',
                'L2s overlap score',
                'L1 trans err',
                'L2 trans err']

    def _check_config_line(self, df: pd.DataFrame,
                                 config_list: list) -> int:
        """
            Take the dataframe and check if the 
            config list is already there. 
            If yes returns the line idx
            If no returns -1 
        """
        config_columns = list(df.columns)[:len(config_list)]

        config_df_values = df[config_columns].values.tolist()

        for idx, row in enumerate(config_df_values):
            if row == config_list:
                return idx
        return -1

    def _fus_dict(self, big_dict: dict, 
                        small_dict: dict) -> dict:
        if len(big_dict.keys()) == 0:
            big_dict = {k: [small_dict[k]] 
                        for k in small_dict.keys()}
        else:
            for k in big_dict.keys():
                big_dict[k].append(small_dict[k])
        return big_dict

    def _get_key_from_config_list(self, config_list):
        key = ''
        for elem in config_list:
            key += str(elem) + '_'
        return key[:-1]







   