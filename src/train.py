"""
    blabla
"""    
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models import BackTranslator
from src.languages import DiscreteSquares, ContinuousSquares
from tools.noise_functions import identity



class BackTranslation:
    """
        blabla
    """       

    def __init__(self, configs: dict) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.configs = configs

        # /!\ Not clean indicate the class if we're in a stochastic
        #     setup (True) or in a deterministic one (False)
        self.stochastic = configs['stochastic']

        # Get things
        self._get_noise_func()
        self._get_criterion()

        # Get the loader
        if configs['name'] == "DiscreteSquares":
            self.languages = DiscreteSquares(configs)
        elif configs['name'] == "ContinuousSquares":
            self.languages = ContinuousSquares(configs)
        else:
            raise Exception('Configs name {} is not implemented !'.format(configs['name']))

        self.L1_train_loader, self.L1_valid_loader, self.L1_test_loader = self.languages.get_dataloaders(1)
        self.L2_train_loader, self.L2_valid_loader, self.L2_test_loader = self.languages.get_dataloaders(2)

        # Build the model
        self._build_model()
        

    def _build_model(self) -> None:
        """
            Build the appropriate model according
            to the configs.
        """
        self.model = BackTranslator(self.configs,
                                    languages = self.languages,
                                    device = self.device,
                                    stochastic = self.stochastic,
                                    train_all = False).to(self.device)

    def _get_criterion(self) -> None:
        """
            Select correct criterion.
            Maybe it will change but for now:
                - if stochastic select CrossEntropy
                - if deterministic select MSE
        """
        if self.stochastic:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def _get_noise_func(self) -> None:
        """
            Select the correct noise function
            according to the configs.
        """
        self.noise_func = identity

    def evaluate_denoising(self, loader: torch.utils.data.DataLoader,
                                 language: int) -> float:
        """
            Evaluate the model performance on the 
            loader.
            /!\ For now only returns loss /!\
        """ 

        self.model.eval()
        eval_loss = 0.

        with torch.no_grad():
            for b in loader:
                
                sentences = b['sentences']
                sentences_noisy = self.noise_func(sentences)
                sentences_noisy = sentences_noisy.to(self.device)

                if self.stochastic:
                    targets = b['sentences_ids'].to(self.device) # If stochastic we feed the criterion
                                                                 # with class ids.
                else:
                    targets = sentences.to(self.device) # If deterministic we feed the criterion
                                                        # directly with the sentences.

                sentences_recons = self.model.denoising_autoencoding(sentences_noisy, 
                                                                     language)

                loss = self.criterion(sentences_recons, targets)
                eval_loss += loss.item()
            
            return np.round(eval_loss / len(loader), 2)

    def evaluate_backtranslation(self, loader1: torch.utils.data.DataLoader,
                                       loader2: torch.utils.data.DataLoader) -> tuple[float, float]:
        """
            Evaluate a model on the test_loader.
            For now returns the loss according to criterion.
        """
        
        self.model.eval()
        eval_loss1, eval_loss2 = 0., 0.
        
        loader_size = min(len(loader1), len(loader2))
        
        iterator1 = iter(loader1)
        iterator2 = iter(loader2)

        with torch.no_grad():
            for k in range(loader_size):
                
                b1 = next(iterator2)
                b2 = next(iterator1)

                sentences1 = b1['sentences']
                sentences1_noisy = self.noise_func(sentences1)
                sentences1_noisy = sentences1_noisy.to(self.device)

                sentences2 = b2['sentences']
                sentences2_noisy = self.noise_func(sentences2)
                sentences2_noisy = sentences2_noisy.to(self.device)

                if self.stochastic:
                    targets1 = b1['sentences_ids'].to(self.device) # If stochastic we feed the criterion
                    targets2 = b2['sentences_ids'].to(self.device) # with class ids.
                else:
                    targets1 = sentences1.to(self.device) # If deterministic we feed the criterion
                    targets2 = sentences2.to(self.device) # directly with the sentences.

                sentences1_recons = self.model.back_translation(sentences1_noisy, 1)
                sentences2_recons = self.model.back_translation(sentences2_noisy, 2)

                loss1 = self.criterion(sentences1_recons, targets1)
                loss2 = self.criterion(sentences2_recons, targets2)
                eval_loss1 += loss1.item()
                eval_loss2 += loss2.item()
            
        return np.round(eval_loss1 / loader_size, 2), np.round(eval_loss2 / loader_size, 2)

    def evaluate(self, loader1: DataLoader, 
                       loader2: DataLoader,
                       denoising: bool,
                       adversial: bool) -> tuple[float, float]:    
        """
            Evaluate the model on the different tasks.
        """

        metrics = {'total': 0.,
                   'BT L1': 0.,
                   'BT L2': 0.,
                   'DAE L1': 0.,
                   'DAE L2': 0.}

        metrics['BT L1'], metrics['BT L2'] = self.evaluate_backtranslation(loader1,
                                                                           loader2)

        if denoising:
            metrics['DAE L1'] = self.evaluate_denoising(loader1, 1)
            metrics['DAE L2'] = self.evaluate_denoising(loader2, 2)

        for k in metrics.keys():
            if k == 'total':
                continue
            metrics['total'] += metrics[k]

        return metrics


    def step_denoising(self, batch, language: int) -> torch.Tensor:
        """
            Execute one step of forward of denoising
            auto-encoding.
        """

        sentences = batch['sentences']
        sentences_noisy = self.noise_func(sentences)
        sentences_noisy = sentences_noisy.to(self.device)

        if self.stochastic:
            targets = batch['sentences_ids'].to(self.device)
        else:
            targets = sentences.to(self.device)

        sentences_recons = self.model.denoising_autoencoding(sentences_noisy,
                                                             language)

        loss = self.criterion(sentences_recons, targets)

        return loss

    def step_backtranslation(self, batch, language: int) -> torch.Tensor:
        """
            Execute one step of forward backtranslation.
        """

        sentences = batch['sentences']
        sentences_noisy = self.noise_func(sentences)
        sentences_noisy = sentences_noisy.to(self.device)

        if self.stochastic:
            targets = batch['sentences_ids'].to(self.device)
        else:
            targets = sentences.to(self.device) 
            
        sentences_recons = self.model.back_translation(sentences_noisy, 
                                                       language)
        
        loss = self.criterion(sentences_recons, targets)

        return loss


    def train(self, denoising: bool = False, # Train with denoising autocencoding objective
                    adversial: bool = False, # Train with adversial objective
                    silent: bool = False) -> None:
        """
            Train the model.
        """

        if not(silent):
            print("############ TRAINING BACKTRANSLATION ############")
            print("\n \t DATASET : {}".format(self.configs['name']))
            print("\n \t denoising : {}".format(denoising))
            print("\t adversial : {}".format(adversial))
        
        loader_size = min(
                        len(self.L1_train_loader), 
                        len(self.L2_train_loader)
                        )

        # We use Adam each time so it is defined here    
        optimizer = optim.Adam(self.model.parameters())
        
        valid_metrics = self.evaluate(self.L1_valid_loader, 
                                      self.L2_valid_loader,
                                      denoising,
                                      adversial)
        
        if not(silent):
            print("\n ### Initial Metrics ### ")
            print("\t Total Loss : {}".format(valid_metrics['total']))
            print('\t BT : L1 Loss {} | L2 Loss {}'.format(valid_metrics['BT L1'],
                                                           valid_metrics['BT L2']))
            if denoising:
                print('\t DAE : L1 Loss {} | L2 Loss {}'.format(valid_metrics['DAE L1'],
                                                                valid_metrics['DAE L2']))
        
        #best_test_loss = valid_loss

        for epoch in range(self.configs['n_epochs']):
            
            self.model.train()
            epoch_loss = 0.

            L1_train_iterator = iter(self.L1_train_loader)
            L2_train_iterator = iter(self.L2_train_loader)

            for _ in range(loader_size):

                L1_batch = next(L1_train_iterator)
                L2_batch = next(L2_train_iterator)
                ###################################
                # sending data to CUDA is not optimized
                # at all right now... TBC
                #################################

                loss = 0. 

                L1_loss_bt = self.step_backtranslation(L1_batch,
                                                       language = 1)
                L2_loss_bt = self.step_backtranslation(L2_batch,
                                                       language = 2)
                loss += L1_loss_bt
                loss += L2_loss_bt

                if denoising:
                    L1_loss_dae = self.step_denoising(L1_batch,
                                                      language = 1)
                    L2_loss_dae = self.step_denoising(L2_batch,
                                                      language = 2)
                    loss += L1_loss_dae
                    loss += L2_loss_dae

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                
            train_loss = np.round(epoch_loss/loader_size, 2)
            valid_metrics = self.evaluate(self.L1_valid_loader,
                                          self.L2_valid_loader,
                                          denoising,
                                          adversial)
            
            if not(silent):
                print("### Epoch {} ###".format(epoch))
                print("\t Total Train Loss : {}  | Valid Loss : {}".format(train_loss, 
                                                                           valid_metrics['total']))
                
                print('\t BT Valid : L1 Loss {} | L2 Loss {}'.format(valid_metrics['BT L1'],
                                                                     valid_metrics['BT L2']))
                                                        
                if denoising:
                    print('\t DAE Valid : L1 Loss {} | L2 Loss {}'.format(valid_metrics['DAE L1'],
                                                                          valid_metrics['DAE L2']))
            
            #best_test_loss = save_weights(auto_encoder, test_loss, best_test_loss, weights_name)

        
