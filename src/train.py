"""
    blabla
"""    
import msilib
import numpy as np
from tools.helpers import get_weights_name
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ray import tune
import wandb

from src.models import BackTranslator, Discriminator, LSTMEnc, LSTMDec
from src.languages import ContinuousSquares, GaussianMixture, WMT14, SimpleEN_FR, PCFG
from tools.noise_functions import Noiser
from tools.helpers import get_weights_name, binary_accuracy, log_sentences_table
from tools.language_similarity import LanguageSim
from analysis.analyser import Analyser2D


class BackTranslation:
    """
        blabla
    """       

    def __init__(self, configs: dict,
                       ray_tune: bool = False) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.configs = configs
        self.ray_tune = ray_tune
        self.metrics_historic = {}

        # /!\ Not clean indicate the class if we're in a stochastic
        #     setup (True) or in a deterministic one (False)
        self.stochastic = configs['training']['stochastic']

        # Get things
        self._get_criterions()

        # Get the loader
        if configs['name'] == "ContinuousSquares":
            self.languages = ContinuousSquares(configs)
        elif configs['name'] == "GaussianMixture":
            self.languages = GaussianMixture(configs)
        elif configs['name'] == "WMT14":
            self.languages = WMT14(configs)
        elif configs['name'] == "SimpleEN_FR":
            self.languages = SimpleEN_FR(configs)
        elif configs['name'] == "PCFG":
            self.languages = PCFG(configs)
        else:
            raise Exception('Configs name {} is not implemented !'.format(configs['name']))

        self.L1_train_loader, self.L1_train_gold_loader, self.L1_valid_loader, self.L1_test_loader = self.languages.get_dataloaders(
                                                                                        language = 1,
                                                                                        batch_size = configs['training']['batch_size'],
                                                                                        prop_gold = configs['training']['prop_gold']
                                                                                        )
        self.L2_train_loader, self.L2_train_gold_loader, self.L2_valid_loader, self.L2_test_loader = self.languages.get_dataloaders(
                                                                                        language = 2,
                                                                                        batch_size = configs['training']['batch_size'],
                                                                                        prop_gold = configs['training']['prop_gold']
                                                                                        )
        L1_train_size, L2_train_size = 0, 0
        if self.L1_train_loader is not None:
            L1_train_size += len(self.L1_train_loader)
        if self.L1_train_gold_loader is not None:
            L1_train_size += len(self.L1_train_gold_loader)
        if self.L2_train_loader is not None:
            L2_train_size += len(self.L2_train_loader)
        if self.L2_train_gold_loader is not None:
            L2_train_size += len(self.L2_train_gold_loader)

        print("L1 loader sizes : {} (train) ; {} (valid) ; {} (test)".format(L1_train_size,
                                                                             len(self.L1_valid_loader),
                                                                             len(self.L1_test_loader)
                                                                             )
                                                                    )
        print("L2 loader sizes : {} (train) ; {} (valid) ; {} (test)".format(L2_train_size,
                                                                             len(self.L2_valid_loader),
                                                                             len(self.L2_test_loader)
                                                                             )
                                                                    )
        """                                                                
        b = next(iter(self.L1_train_loader))
        print(b['sentences'].shape)
        print(b['sentences'][0])
        print([self.languages.tokenizer.id_to_token(token) for token in b['sentences'][0]])
        print([self.languages.tokenizer.id_to_token(token) for token in b['gold translation'][0]])
        print(b['lengths'])
        exit(0)
        #TEST
        encoder = LSTMEnc(configs)
        decoder = LSTMDec(configs, encoder)

        inputs = {'x': b['sentences'],
                  'lengths': b['lengths']}

        encoded = encoder(inputs, 1)
        scores = decoder(encoded, b['sentences'], 1)
        sent_gen = decoder.generate(encoded, language = 1, sample = False, temperature = None)

        print("Scores : {}".format(scores.shape))
        print(sent_gen)
        print(sent_gen[0].shape)

        exit(0)
        """
        # Build the model
        self._build_models()
        
        self._get_optimizers()

        self._get_language_similarity()

        # Total Steps Counter
        self.total_steps = 0

        # Noise
        
        self._get_noise_func()

        # Analyser
        if configs['name'] in ['GaussianMixture', 'ContinuousSquares']:
            self.analyser = Analyser2D(configs,
                                    load_weights = False,
                                    L1_test_loader = self.L1_test_loader,
                                    L2_test_loader = self.L2_test_loader,
                                    model = self.model,
                                    M_discriminator = self.M_discriminator)
        

    def _build_models(self) -> None:
        """
            Build the appropriate models according
            to the configs.
        """
        self.model = BackTranslator(self.configs,
                                    device = self.device,
                                    stochastic = self.stochastic,
                                    train_all = False).to(self.device)

        if self.configs['training']['adversial']:
            self.M_discriminator = Discriminator(
                            self.configs['models']['discriminator']
                            ).to(self.device)
        else:
            self.M_discriminator = None

    def _get_criterions(self) -> None:
        """
            Select correct criterion.
            Maybe it will change but for now:
                - if stochastic select CrossEntropy
                - if deterministic select MSE
        """
        if self.stochastic:
            self.criterion = nn.CrossEntropyLoss(
                        ignore_index=self.configs['dataset']['pad_index']
                        )
        else:
            self.criterion = nn.MSELoss()

        if self.configs['training']['adversial']:
            self.adversial_criterion = nn.BCEWithLogitsLoss()

    def _get_optimizers(self):
        # We use Adam each time so it is defined here    
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr = self.configs['training']['lr'])

        # Maybe                        
        if self.configs['training']['adversial']:
            self.discriminator_optimizer = optim.Adam(self.M_discriminator.parameters(),
                                                      lr = self.configs['training']['lr adversial'], 
                                                      betas=(0.5, 0.999))
            self.train_D = True
            self.train_G = False

    def _get_language_similarity(self):
        # Measuring similarity between sentences
        # from a language. For now MSE but with
        # real life example BLUE score or something..
        if self.configs['dataset']['language_similarity'] != -1:
            self.language_similarity = LanguageSim(
                    self.configs['dataset']['language_similarity'],
                    tokenizer = self.languages.tokenizer
                    )
        else:
            self.language_similarity = None

    def _get_noise_func(self) -> None:
        """
            Select the correct noise function
            according to the configs.
        """
        
        self.noiser = Noiser(noise_func = self.configs['training']['noise_function'],
                             noise_intensity = self.configs['training']['noise_intensity'],
                             tokenizer = self.languages.tokenizer)

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
                sentences_noisy, lengths = self.noiser.noise(
                                                inputs = {'x': sentences,
                                                          'lengths': b['lengths'],
                                                          'language': language}
                                                )
                sentences_noisy = sentences_noisy.to(self.device)

                targets = sentences.to(self.device) 
                if self.stochastic:
                    targets = targets[:,1:].to(torch.long) # We do not feed BOS to loss

                # Create the forward pass inputs dict
                inputs = {'x': sentences_noisy,
                          'lengths': lengths}

                # Forward pass
                sentences_recons = self.model.denoising_autoencoding(inputs, 
                                                                     language)

                loss = self.criterion(sentences_recons, targets)
                eval_loss += loss.item()
            
            return np.round(eval_loss / len(loader), 2)

    def evaluate_translation(self, loader1: torch.utils.data.DataLoader,
                                   loader2: torch.utils.data.DataLoader) -> torch.Tensor:
        """
            Function that evaluate the Trad perf.
            Compute MSE Loss between y and Trad(x).
        """
        
        self.model.eval()
        eval_err_1, eval_err_2 = 0., 0.
        eval_sim_1, eval_sim_2 = 0., 0.
        
        loader_size = min(len(loader1), len(loader2))
        
        iterator1 = iter(loader1)
        iterator2 = iter(loader2)

        # In case of logging, random batch idx to log
        batch_idx = np.random.randint(loader_size)

        with torch.no_grad():
            for k in range(loader_size):
                
                b1 = next(iterator1)
                b2 = next(iterator2)

                sentences1 = b1['sentences'].to(self.device)
                sentences2 = b2['sentences'].to(self.device)

                # Create the forward pass inputs dict
                inputs1 = {'x': sentences1,
                           'lengths': b1['lengths']}
                inputs2 = {'x': sentences2,
                           'lengths': b2['lengths']}

                if self.configs['dataset']['only_gold_translation']:
                    inputs1['gold translation'] =  b1['gold translation'].to(self.device)
                    inputs2['gold translation'] =  b2['gold translation'].to(self.device)

                # Getting the gold value
                if self.configs['dataset']['only_gold_translation']:
                    sentences1_translation_gold = b1['gold translation'][:,1:].to(self.device).to(torch.long) # Do not take BOS token into loss
                    sentences2_translation_gold = b2['gold translation'][:,1:].to(self.device).to(torch.long) # Doesn't change anything for BLEU
                else:
                    sentences1_translation_gold = self.model.translate(inputs1, 
                                                                       1,
                                                                       True)
                    sentences2_translation_gold = self.model.translate(inputs2, 
                                                                       2,
                                                                       True)

                # Forward pass
                sentences1_translation, scores1_translation = self.model.translate(
                                                                inputs1, 
                                                                1,
                                                                output_scores = True
                                                                )
                sentences2_translation, scores2_translation = self.model.translate(
                                                                inputs2, 
                                                                2,
                                                                output_scores = True
                                                                )

                # Language Similarity
                if self.language_similarity is not None:
                    self.language_similarity.add_batch(
                                            sentences1_translation, 
                                            sentences1_translation_gold,
                                            language = 1
                                            )
                    self.language_similarity.add_batch(
                                            sentences2_translation, 
                                            sentences2_translation_gold,
                                            language = 2
                                            )

                # Translation Error
                loss1 = self.criterion(scores1_translation, 
                                       sentences1_translation_gold)
                loss2 = self.criterion(scores2_translation, 
                                       sentences2_translation_gold)
                eval_err_1 += loss1.item()
                eval_err_2 += loss2.item()

                # Log wandb
                if self.configs['training']['wandb']:
                    if k == batch_idx:
                        log_sentences_table(sentences1, 
                                            sentences1_translation,
                                            inputs1['gold translation'],
                                            self.languages.tokenizer,
                                            name = "from {} to {}".format(
                                                        self.configs['dataset']['L1']['name'],
                                                        self.configs['dataset']['L2']['name'],
                                            ))
                        log_sentences_table(sentences2, 
                                            sentences2_translation,
                                            inputs2['gold translation'],
                                            self.languages.tokenizer,
                                            name = "from {} to {}".format(
                                                        self.configs['dataset']['L2']['name'],
                                                        self.configs['dataset']['L1']['name'],
                                            ))

                

        if self.language_similarity is not None:
            eval_sim_1 = self.language_similarity.compute(
                                            language = 1
                                            )
            eval_sim_2 = self.language_similarity.compute(
                                            language = 2
                                            )    
            return (np.round(eval_err_1 / loader_size, 2), 
                    np.round(eval_err_2 / loader_size, 2), 
                    np.round(eval_sim_1, 2), 
                    np.round(eval_sim_2, 2))
        return (np.round(eval_err_1 / loader_size, 2), 
                np.round(eval_err_2 / loader_size, 2), 
                None,
                None)

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
                
                b1 = next(iterator1)
                b2 = next(iterator2)

                sentences1 = b1['sentences']
                sentences1_noisy, lengths1 = self.noiser.noise(
                                                inputs = {'x': sentences1,
                                                          'lengths': b1['lengths'],
                                                          'language': 1}
                                                )
                sentences1_noisy = sentences1_noisy.to(self.device)

                sentences2 = b2['sentences']
                sentences2_noisy, lengths2 = self.noiser.noise(
                                                inputs = {'x': sentences2,
                                                          'lengths': b2['lengths'],
                                                          'language': 2}
                                                )
                sentences2_noisy = sentences2_noisy.to(self.device)

                # In backtranslation targets is the input sentence
                targets1 = sentences1.to(self.device)
                targets2 = sentences2.to(self.device)
                if self.stochastic:
                    targets1 = targets1[:,1:].to(torch.long)
                    targets2 = targets2[:,1:].to(torch.long)

                # Create the forward pass inputs dict
                inputs1 = {'x': sentences1_noisy,
                           'lengths': lengths1}
                inputs2 = {'x': sentences2_noisy,
                           'lengths': lengths2}

                # Forward Pass
                sentences1_recons = self.model.back_translation(inputs1, 1)
                sentences2_recons = self.model.back_translation(inputs2, 2)

                # Loss
                loss1 = self.criterion(sentences1_recons, targets1)
                loss2 = self.criterion(sentences2_recons, targets2)
                eval_loss1 += loss1.item()
                eval_loss2 += loss2.item()
            
        return np.round(eval_loss1 / loader_size, 2), np.round(eval_loss2 / loader_size, 2)

    def evaluate_adversial(self, loader1: torch.utils.data.DataLoader,
                                 loader2: torch.utils.data.DataLoader) -> tuple[float, float]:
        """
            Evaluate the discriminator on both language.
            Returns loss (BCE) and accuracy.
            For now only evaluate the M-discriminator.
            Might be interesting to use a L-discriminator.
        """
        
        self.M_discriminator.eval()
        dis_loss = 0.
        dis_acc = 0.
        
        loader_size = min(len(loader1), len(loader2))
        
        iterator1 = iter(loader1)
        iterator2 = iter(loader2)

        with torch.no_grad():
            for k in range(loader_size):
                
                b1 = next(iterator1)
                b2 = next(iterator2)

                sentences1 = b1['sentences'].to(self.device)
                sentences2 = b2['sentences'].to(self.device)

                # Create the forward pass inputs dict
                inputs1 = {'x': sentences1,
                           'lengths': b1['lengths']}
                inputs2 = {'x': sentences2,
                           'lengths': b2['lengths']}

                # Encode in latent space
                m1s = self.model.r(inputs1, 1)
                m2s = self.model.r(inputs2, 2)
                ### m1s will be dict too so things will change here

                # predict
                L1_logits = self.M_discriminator(m1s)
                L2_logits = self.M_discriminator(m2s)

                targets1 = torch.zeros_like(L1_logits).to(self.device)
                targets2 = torch.ones_like(L2_logits).to(self.device)

                loss1 = self.adversial_criterion(L1_logits, 
                                                 targets1)
                loss2 = self.adversial_criterion(L2_logits, 
                                                 targets2)

                acc1 = binary_accuracy(L1_logits, targets1)
                acc2 = binary_accuracy(L2_logits, targets2)

                dis_loss += loss1.item() + loss2.item()
                dis_acc += 0.5*(acc1.item() + acc2.item())
            
        return np.round(dis_loss / loader_size, 2), np.round(dis_acc / loader_size, 2)

    def evaluate(self, loader1: DataLoader, 
                       loader2: DataLoader,
                       ) -> tuple[float, float]:    
        """
            Evaluate the model on the different tasks.
        """

        metrics = {'total': 0.,
                   'BT L1': 0.,
                   'BT L2': 0.,
                   'Trans Err L1': 0.,
                   'Trans Err L2': 0.
                   }

        metrics['BT L1'], metrics['BT L2'] = self.evaluate_backtranslation(loader1,
                                                                           loader2)
        translation_res = self.evaluate_translation(loader1,
                                                    loader2)

        metrics['Trans Err L1'] = translation_res[0]
        metrics['Trans Err L2'] = translation_res[1]
        if translation_res[2] is not None:
            metrics['Trans Sim L1'] = translation_res[2]
            metrics['Trans Sim L2'] = translation_res[3]
        

        if self.configs['training']['denoising']:
            metrics['DAE L1'] = self.evaluate_denoising(loader1, 1)
            metrics['DAE L2'] = self.evaluate_denoising(loader2, 2)
        if self.configs['training']['adversial']:
            loss_dis, acc_dis = self.evaluate_adversial(loader1,
                                                        loader2)
            metrics['D Loss'] = loss_dis
            metrics['D Acc'] = acc_dis

        for k in metrics.keys():
            if k in ['total', 
                     'Trans Err L1', 
                     'Trans Err L2',
                     'Trans Sim L1', 
                     'Trans Sim L2',
                     'D Loss',
                     'D Acc']:
                continue
            metrics['total'] += metrics[k]

        if self.metrics_historic:
            for k in metrics.keys():
                self.metrics_historic[k].append(
                                        metrics[k]
                                        )
        else:
            for k in metrics.keys():
                self.metrics_historic[k] = [metrics[k]]


        return metrics


    def step_denoising(self, batch, language: int) -> torch.Tensor:
        """
            Execute one step of forward of denoising
            auto-encoding.
        """

        sentences = batch['sentences']
        sentences_noisy, lengths = self.noiser.noise(
                                inputs = {'x': sentences,
                                          'lengths': batch['lengths'],
                                          'language': language}
                                )
        sentences_noisy = sentences_noisy.to(self.device)

        targets = sentences.to(self.device)
        if self.stochastic:
            targets = targets[:,1:].to(torch.long)

        # Create the forward pass inputs dict
        inputs = {'x': sentences_noisy,
                  'lengths': lengths}

        # Forward pass
        sentences_recons = self.model.denoising_autoencoding(inputs,
                                                             language)
        # Loss
        loss = self.criterion(sentences_recons, targets)

        return loss

    def step_adversial_dis(self, L1_batch, 
                                 L2_batch,
                                 train_D,
                                 train_G) -> torch.Tensor:
        """
            Execute one step of adversial training, including the backward pass.
        """
        ### Discriminator ###
        self.M_discriminator.train()

        # Important
        self.M_discriminator.zero_grad()
        
        ### data equiv ###
        L2_sentences = L2_batch['sentences'].to(self.device)
        # Create the forward pass inputs dict
        inputs2 = {'x': L2_sentences,
                   'lengths': L2_batch['lengths']}
        # Generate "real" ms batch with G2
        # Again ms will be dict
        m2s = self.model.r(inputs2, 2)
        targets2 = torch.ones(m2s.shape[0]).to(self.device)*0.9 # *0.9 for stability

        if train_D:
            # Forward pass real batch through D
            dis_L2_logits = self.M_discriminator(m2s.detach())
            # Calculate loss on all-real batch
            dis_loss2 = self.adversial_criterion(dis_L2_logits, 
                                                 targets2)
            # Calculate gradients for D in backward pass
            dis_loss2.backward()

        ### noise equiv ###
        # format batch
        L1_sentences = L1_batch['sentences'].to(self.device)
        # Create the forward pass inputs dict
        inputs1 = {'x': L1_sentences,
                   'lengths': L1_batch['lengths']}
        # Generate "fake" ms batch with G1
        m1s = self.model.r(inputs1, 1)
        targets1 = torch.ones(m1s.shape[0]).to(self.device)*0.1 # *0.1 for stability

        if train_D:
            # Classify all fake batch with D
            dis_L1_logits = self.M_discriminator(m1s.detach())
            # Calculate D's loss on the all-fake batch
            
            dis_loss1 = self.adversial_criterion(dis_L1_logits, 
                                                targets1)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            dis_loss1.backward()

            # D's optimizer step
            self.discriminator_optimizer.step()
        
        if train_G:
            ### Generator ###
            self.M_discriminator.eval()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            dis_L2_logits = self.M_discriminator(m2s)
            dis_L1_logits = self.M_discriminator(m1s)

            # Discriminator Loss (to update the discriminator)
            loss1 = self.adversial_criterion(dis_L1_logits, 
                                             targets2)
            loss2 = self.adversial_criterion(dis_L2_logits, 
                                             targets1)

            return loss1 + loss2
        else:
            return 0.

    def step_backtranslation(self, batch, 
                                   language: int, 
                                   gold: bool = False) -> torch.Tensor:
        """
            Execute one step of forward backtranslation.
        """

        sentences = batch['sentences']
        sentences_noisy, lengths = self.noiser.noise(
                                inputs = {'x': sentences,
                                          'lengths': batch['lengths'],
                                          'language': language}
                                )
        sentences_noisy = sentences_noisy.to(self.device)

        targets = sentences.to(self.device)
        if self.stochastic:
                targets = targets[:,1:].to(torch.long)

        # Create the forward pass inputs dict
        inputs = {'x': sentences_noisy,
                  'lengths': lengths}
        if gold and self.configs['dataset']['only_gold_translation']:
            inputs['gold translation'] = batch['gold translation'].to(self.device)
            inputs['gold lengths'] = batch['gold lengths']

        # Forward pass    
        sentences_recons = self.model.back_translation(inputs, 
                                                       language,
                                                       gold)
        
        loss = self.criterion(sentences_recons, 
                              targets)

        return loss


    def train(self, silent: bool = False,
                    analyse: bool = True) -> None:
        """
            Train the model.
        """
        if self.configs['training']['wandb']:
            wandb.init(
                # Set the project where this run will be logged
                project="Unsupervised Machine Translation", 
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=get_weights_name(self.configs)[12:-3], 
                # Track hyperparameters and run metadata
                config={
                "learning_rate": self.configs['training']['lr'],
                "batch_size": self.configs['training']['batch_size'],
                "prop_gold": self.configs['training']['prop_gold'],
                "encoder": self.configs['models']['encoder']['model'],
                "decoder": self.configs['models']['decoder']['model'],
                "dataset": self.configs['name'],
                "epochs": self.configs['training']['n_epochs'],
                })

        if not(silent):
            print("\n ############ TRAINING BACKTRANSLATION ############")
            print("\n \t DATASET : {}".format(
                                        self.configs['name']
                                        ))
            print("\n \t prop supervised : {}".format(
                                        self.configs['training']['prop_gold']
                                        ))
            print("\t denoising : {}".format(
                                        self.configs['training']['denoising']
                                        ))
            print("\t adversial : {}".format(
                                        self.configs['training']['adversial']
                                        ))
            print("\t LM pretraining : {}".format(
                                        self.configs['training']['lm_pretraining']
                                        ))
            print("\t shared encoder : {}".format(
                                        self.configs['training']['share_enc']
                                        ))
            print("\t shared decoder : {}".format(
                                        self.configs['training']['share_dec']
                                        ))
            print("\t early stopping : {}".format(
                                        self.configs['training']['early_stopping']
                                        ))
            print("\t wandb : {}".format(
                                        self.configs['training']['wandb']
                                        ))
        
        valid_metrics = self.evaluate(self.L1_valid_loader, 
                                      self.L2_valid_loader
                                     )
        if self.configs['training']['early_stopping']:
            early_stopping_counter = 0
        prev_loss = valid_metrics['total']
        best_loss = valid_metrics['total']
        
        if not(silent):
            print("\n ### Initial Metrics ### ")
            print("\t Total Loss : {}".format(
                                            valid_metrics['total']
                                            ))
            print('\t BT : L1 Loss {} | L2 Loss {}'.format(
                                            valid_metrics['BT L1'],
                                            valid_metrics['BT L2']
                                            ))
            if self.configs['training']['denoising']:
                print('\t DAE : L1 Loss {} | L2 Loss {}'.format(
                                            valid_metrics['DAE L1'],
                                            valid_metrics['DAE L2']
                                            ))
            if self.configs['training']['adversial']:
                print('\t Discriminator : Loss {} | Acc. {}'.format(
                                            valid_metrics['D Loss'],
                                            valid_metrics['D Acc']
                                            ))
            print('\t L1 trans. err. {} | L2 trans. err.  {}'.format(
                                            valid_metrics['Trans Err L1'],
                                            valid_metrics['Trans Err L2']
                                            ))
            if self.language_similarity is not None:
                print('\t L1 {} {} | L2 {} {}'.format(
                                            self.configs['dataset']['language_similarity'].upper(),
                                            valid_metrics['Trans Sim L1'],
                                            self.configs['dataset']['language_similarity'].upper(),
                                            valid_metrics['Trans Sim L2']
                                            ))
        
        if self.configs['training']['wandb']:
            wandb.log({**valid_metrics})

        if self.configs['training']['lm_pretraining']:
            print("\n ### LM PRETRAINING ### ")
            self._lm_pretraining(silent)
        
        #best_test_loss = valid_loss
        print("\n ### BACKTRANSLATION ### ")
        for epoch in range(self.configs['training']['n_epochs']):
            train_loss, valid_metrics = self._train_one_epoch(analyse)

            if self.configs['training']['wandb']:
                wandb.log({**valid_metrics})
            
            if not(silent):
                print("### Epoch {}/{} ###".format(epoch+1,
                                                   self.configs['training']['n_epochs']))
                print("\t Total Train Loss : {}  | Valid Loss : {}".format(
                                                                train_loss, 
                                                                valid_metrics['total']
                                                                ))
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
                                                                valid_metrics['Trans Err L1'],
                                                                valid_metrics['Trans Err L2']
                                                                ))
                
                if self.language_similarity is not None:
                    print('\t L1 {} {} | L2 {} {}'.format(
                                                self.configs['dataset']['language_similarity'].upper(),
                                                valid_metrics['Trans Sim L1'],
                                                self.configs['dataset']['language_similarity'].upper(),
                                                valid_metrics['Trans Sim L2']
                                                ))

            if self.configs['training']['early_stopping']:
                if early_stopping_counter == self.configs['training']['early_stopping_patience']:
                    print("Early stopping triggered. Stopping training...")
                    break
                
                if abs(valid_metrics['total'] - prev_loss) < 0.005 or (valid_metrics['total'] > prev_loss):
                    print('Early stoping step {}/{}'.format(
                                                    early_stopping_counter, 
                                                    self.configs['training']['early_stopping_patience']
                                                    ))
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0

            prev_loss = valid_metrics['total']

                
        
            if (valid_metrics['total'] < best_loss) and not(self.ray_tune): # It seems that ray tune breaks here
                
                print("Saving weights...")
                
                best_loss = valid_metrics['total']
                # Save weights
                weights_name = get_weights_name(self.configs)
            
                torch.save(self.model.state_dict(), weights_name)

                if self.configs['training']['adversial']:
                    D_weights_name = weights_name[:-3] + '_disc.pt' 
                    torch.save(self.M_discriminator.state_dict(), 
                            D_weights_name)

        if self.configs['training']['wandb']:
            wandb.finish()

    
    def _train_one_epoch(self, analyse):
        epoch_loss = 0.
        loader_size = 0

        # Not fully supervised training
        if self.configs['training']['prop_gold'] < 1:
            L1_train_iterator = iter(self.L1_train_loader)
            L2_train_iterator = iter(self.L2_train_loader)

            loader_size += min(
                            len(self.L1_train_loader), 
                            len(self.L2_train_loader)
                            )

        #print("no gold ", loader_size)

        # Not fully unsupervised training
        if self.configs['training']['prop_gold'] > 0:

            L1_train_gold_iterator = iter(self.L1_train_gold_loader)
            L2_train_gold_iterator = iter(self.L2_train_gold_loader)

            loader_gold_size =  min(
                                    len(self.L1_train_gold_loader), 
                                    len(self.L2_train_gold_loader)
                                    )

            #print("gold ", loader_gold_size)

            golds = np.array(
                            [True for _ in range(loader_gold_size)] +\
                            [False for _ in range(loader_size)]
                        )
            np.random.shuffle(golds)

            loader_size += loader_gold_size

        for k in tqdm.tqdm(range(loader_size)):
            self.model.train()

            loss = 0.

            # To simulate a supervised step
            gold = False
            if self.configs['training']['prop_gold'] > 0:
                gold = golds[k]

            if gold:
                L1_batch = next(L1_train_gold_iterator)
                L2_batch = next(L2_train_gold_iterator)
            else:
                L1_batch = next(L1_train_iterator)
                L2_batch = next(L2_train_iterator)
                
            L1_loss_bt = self.step_backtranslation(L1_batch,
                                                   language = 1,
                                                   gold = gold)
            L2_loss_bt = self.step_backtranslation(L2_batch,
                                                   language = 2,
                                                   gold = gold)
            loss += L1_loss_bt
            loss += L2_loss_bt
                

            if self.configs['training']['denoising']:
                L1_loss_dae = self.step_denoising(L1_batch,
                                                  language = 1)
                L2_loss_dae = self.step_denoising(L2_batch,
                                                  language = 2)
                loss += L1_loss_dae
                loss += L2_loss_dae
            if self.configs['training']['adversial']:
                loss_adv = self.step_adversial_dis(L1_batch,
                                                   L2_batch,
                                                   self.discriminator_optimizer,
                                                   self.train_D,
                                                   self.train_G)
                loss += loss_adv

                if (self.total_steps + 1)%self.configs['training']['adversial_patience'] == 0:
                    self.train_D = not(self.train_D)
                    self.train_G = not(self.train_G)

                    
            # Could compensate Gold Proportion
            if gold:
                #loss *= 1./self.configs['training']['prop_gold']
                pass
            ##

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            self.total_steps += 1

            if analyse and self.total_steps%self.configs['training']['analyse_every'] == 0:
                self.evaluate(
                        self.L1_valid_loader,
                        self.L2_valid_loader
                        )
                self.analyser.L_and_M_stats(n_batches = 8,
                                            metrics_historic=self.metrics_historic,
                                            plot = False,
                                            save = True)

        
        train_loss = np.round(epoch_loss/loader_size, 2)

        valid_metrics = self.evaluate(
                                self.L1_valid_loader,
                                self.L2_valid_loader
                                )
        # Possibly remove
        #if self.ray_tune:
        #    tune.report(loss = valid_metrics['total'])
        
        return train_loss, valid_metrics


    def _lm_pretraining(self, silent: bool = False):
            
            for epoch in range(self.configs['training']['pretraining_n_epochs']):
            
                epoch_loss = 0.
                loader_size = 0

                # Not fully supervised training
                if self.configs['training']['prop_gold'] < 1:
                    L1_train_iterator = iter(self.L1_train_loader)
                    L2_train_iterator = iter(self.L2_train_loader)

                    loader_size += min(
                                    len(self.L1_train_loader), 
                                    len(self.L2_train_loader)
                                    )

                # Not fully unsupervised training
                if self.configs['training']['prop_gold'] > 0:

                    L1_train_gold_iterator = iter(self.L1_train_gold_loader)
                    L2_train_gold_iterator = iter(self.L2_train_gold_loader)

                    loader_gold_size =  min(
                                            len(self.L1_train_gold_loader), 
                                            len(self.L2_train_gold_loader)
                                            )

                    golds = np.array(
                                    [True for _ in range(loader_gold_size)] +\
                                    [False for _ in range(loader_size)]
                                )
                    np.random.shuffle(golds)

                    loader_size += loader_gold_size

                for k in tqdm.tqdm(range(loader_size)):
                    self.model.train()

                    loss = 0.

                    # To simulate a supervised step
                    gold = False
                    if self.configs['training']['prop_gold'] > 0:
                        gold = golds[k]

                    if gold:
                        L1_batch = next(L1_train_gold_iterator)
                        L2_batch = next(L2_train_gold_iterator)
                    else:
                        L1_batch = next(L1_train_iterator)
                        L2_batch = next(L2_train_iterator)
                        
                    L1_loss_dae = self.step_denoising(L1_batch,
                                                    language = 1)
                    L2_loss_dae = self.step_denoising(L2_batch,
                                                    language = 2)
                    loss += L1_loss_dae
                    loss += L2_loss_dae

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    self.total_steps += 1
                
                train_loss = np.round(epoch_loss/loader_size, 2)

                valid_loss_denoising_1 = self.evaluate_denoising(
                                                self.L1_valid_loader, 
                                                1)
                valid_loss_denoising_2 = self.evaluate_denoising(
                                                self.L2_valid_loader, 
                                                2)

                if not(silent):
                    print("### Epoch {}/{} ###".format(epoch+1,
                                                       self.configs['training']['pretraining_n_epochs']))
                    print("\t Total Train Loss : {} ".format(
                                                        train_loss
                                                        ))
                    print('\t DAE Valid : L1 Loss {} | L2 Loss {}'.format(
                                                                valid_loss_denoising_1,
                                                                valid_loss_denoising_2
                                                                ))
                                    
            return

        


        
