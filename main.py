

import argparse


from src.train import BackTranslation

if __name__ == '__main__':
    
    """
    # To be changed
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='',
                        dest='dataset',
                        help='Name of the dataset used to evaluate BERT knowledge. Choices : custom, wordnet, trex, bless, evil_trex.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='bert',
                        dest='model',
                        help='Name of the model we use. Choices : bert, bert-cased, bert-large, bert-untrained, distilbert, roberta.')
    parser.add_argument("--content", 
                        help="Compute everything for the content words.",
                        action="store_true")
    parser.add_argument("--logical", 
                        help="Compute everything for the logical words.",
                        action="store_true")
    args = parser.parse_args()
    
    pre_trained_model_name = 'bert-base-uncased'
    dataset_name = args.dataset
    model_name = args.model
    filtration_type = args.filtration_type
    """

    configs_s = {'name': 'DiscreteSquares',
              'stochastic': True,
              'n_sentences': 50000,
              'L_x_min': -1,
              'L_x_max': 5,
              'L_y_min': -1,
              'L_y_max': 1,
              'discrete_step': 0.5, 
              'L1_x_min': -1,
              'L1_x_max': 1,
              'L1_y_min': -1,
              'L1_y_max': 1,
              'L2_x_min': 3,
              'L2_x_max': 5,
              'L2_y_min': -1,
              'L2_y_max': 1,
              'dim_M': 3,
              'dim_L': 3,
              'epsilon1': 0.5,
              'epsilon2': 0.5,
              'max_k': 1, # n in the formula above so z belons to [-4*eps, 4*eps]
              'enc_model': 'linear',
              'dec_model': 'linear',
              'noise_intensity': 0.005,
              'batch_size': 16,
              'n_epochs': 10}

    # To put elsewhere
    Nx_L = int((configs_s['L_x_max'] - configs_s['L_x_min'])/configs_s['discrete_step']) + 1
    Ny_L = int((configs_s['L_y_max'] - configs_s['L_y_min'])/configs_s['discrete_step']) + 1
    Nz_L = 2*configs_s['max_k'] + 1
    dec_dim_out = Nx_L * Ny_L * Nz_L
    configs_s['dec_dim_out'] = dec_dim_out

    configs_d = {'name': 'ContinuousSquares',
              'stochastic': False,
              'n_sentences': 50000,
              'L1_x_min': -1,
              'L1_x_max': 1,
              'L1_y_min': -1,
              'L1_y_max': 1,
              'L2_x_min': 3,
              'L2_x_max': 5,
              'L2_y_min': -1,
              'L2_y_max': 1,
              'dim_M': 2,
              'dim_L': 2,
              'dec_dim_out': 2,
              'enc_model': 'linear',
              'dec_model': 'linear',
              'noise_intensity': 0.005,
              'batch_size': 16,
              'n_epochs': 10}

    configs = configs_s

    # Training
    bt = BackTranslation(configs)
    bt.train(denoising = True,
             silent = False)

        

        
        
