"""
    blabla
"""
from distutils.command.config import config
import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

from src.models import BackTranslator, Discriminator
from src.languages import ContinuousSquares, GaussianMixture
from tools.helpers import get_weights_name, PolyArea, binary_accuracy
from tools.convex_polygon_intersection import intersect

class Analyser2D:
    """
        blablabla I plot stuff blablabla
        /!\ In 2D /!\
    """
    def __init__(self, configs: dict, # Needed to load model, languages, ...
                       load_weights: bool = True,
                       L1_test_loader = None,
                       L2_test_loader = None,
                       model = None,
                       M_discriminator = None) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.configs = configs

        # /!\ Not clean indicate the class if we're in a stochastic
        #     setup (True) or in a deterministic one (False)
        self.stochastic = configs['training']['stochastic']

        # Stats
        self.stats = {}

        # Imgs folder
        self.saving_root = "analysis\\imgs"
        self.imgs_saved = 0

        if (L1_test_loader is not None) and (L2_test_loader is not None):
            print("(analyser) Given test loaders!")
            self.L1_test_loader = L1_test_loader
            self.L2_test_loader = L2_test_loader
        else:
            print("(analyser) No given test loaders. Creating them...")
            # Get the loader
            if configs['name'] == "ContinuousSquares":
                self.languages = ContinuousSquares(configs['dataset'])
            elif configs['name'] == "GaussianMixture":
                self.languages = GaussianMixture(configs['dataset'])
            else:
                raise Exception('Configs name {} is not implemented !'.format(configs['name']))

            _, _, _, self.L1_test_loader = self.languages.get_dataloaders(
                                                                    language = 1,
                                                                    batch_size = configs['training']['batch_size'],
                                                                    prop_gold = configs['training']['prop_gold']
                                                                    )
            _, _, _, self.L2_test_loader = self.languages.get_dataloaders(
                                                                    language = 2,
                                                                    batch_size = configs['training']['batch_size'],
                                                                    prop_gold = configs['training']['prop_gold'])

        # Build the model
        self._load_model(weights = load_weights,
                         model = model,
                         M_discriminator = M_discriminator)


    def _load_model(self, weights : bool = True,
                          model = None,
                          M_discriminator = None):
        """
            Load model and its weights.
        """

        if model is not None:
            print("(analyser) Given model!")
            self.model = model
        else:
            self.model =  BackTranslator(self.configs,
                                        device = self.device,
                                        stochastic = self.stochastic,
                                        train_all = False).to(self.device)
        self.untrained_model =  BackTranslator(self.configs,
                                               device = self.device,
                                               stochastic = self.stochastic,
                                               train_all = False).to(self.device)

        if self.configs['training']['adversial']:
            if M_discriminator is not None:
                self.M_discriminator = M_discriminator
            else:
                self.M_discriminator = Discriminator(
                                self.configs['models']['discriminator']
                                ).to(self.device)
            self.M_discriminator.eval()

        # eval
        self.model.eval()
        self.untrained_model.eval()
        

        if weights:
            weights_name = get_weights_name(self.configs)
            print("\nloading {}...".format(weights_name))
            self.model.load_state_dict(torch.load(weights_name))

            if self.configs['training']['adversial']:
                D_weights_name = weights_name[:-3] + '_disc.pt' 
                self.M_discriminator.load_state_dict(torch.load(D_weights_name))

    def L_and_M_stats(self, n_batches: int = 1,
                            metrics_historic: dict = {},
                            plot: bool = True,
                            save: bool = False) -> None:
        """
            Plot sample from L and their images by the
            encoders of the model.
        """

        L1_test_iterator = iter(self.L1_test_loader)
        L2_test_iterator = iter(self.L2_test_loader)

        L1_samples, L2_samples = [], []
        L1_samples_trans, L2_samples_trans = [], []
        untrained_M1_samples, untrained_M2_samples = [], []
        gold_M1_samples, gold_M2_samples = [], []
        M1_samples, M2_samples = [], []
        M2_samples_from_gold = []
        L1_samples_trans_gold = []
        L2_samples_trans_gold = []
        D_true, D_pred = [], []

        trans_err_L1, trans_err_L2 = 0., 0.

        D_acc = 0.

        # Measuring similarity between sentences
        # from a language. For now MSE but with
        # real life example BLUE score or something..
        language_similarity = nn.MSELoss()


        for _ in range(n_batches):
            try:
                L1_batch = next(L1_test_iterator)
                L2_batch = next(L2_test_iterator)
            except:
                break # end of iteration

            L1_sentences = L1_batch['sentences']
            L2_sentences = L2_batch['sentences']

            with torch.no_grad():
                untrained_L1_ms = self.untrained_model.r(L1_sentences.to(self.device), 1)
                untrained_L2_ms = self.untrained_model.r(L2_sentences.to(self.device), 2)

                gold_L1_ms = self.untrained_model.gold_enc(L1_sentences.to(self.device),
                                                           1)
                gold_L2_ms = self.untrained_model.gold_enc(L2_sentences.to(self.device),
                                                           2)

                L1_ms = self.model.r(L1_sentences.to(self.device), 1)
                L2_ms = self.model.r(L2_sentences.to(self.device), 2)

                ### Discriminator ###
                if self.configs['training']['adversial']:
                    dis_L1_logits = self.M_discriminator(L1_ms)
                    targets1 = torch.zeros_like(dis_L1_logits)
                    dis_L2_logits = self.M_discriminator(L2_ms)
                    targets2 = torch.ones_like(dis_L2_logits)

                    D_acc1 = binary_accuracy(dis_L1_logits, targets1)
                    D_acc2 = binary_accuracy(dis_L2_logits, targets2)
                    D_acc += 0.5*(D_acc1.item() + D_acc2.item())

                ###

                L1_sentences_trans = self.model.translate(L1_sentences.to(self.device), 1)
                L2_sentences_trans = self.model.translate(L2_sentences.to(self.device), 2)

                L1_sentences_trans_gold = self.model.translate(L1_sentences.to(self.device), 
                                                               language = 1,
                                                               gold = True)
                L2_sentences_trans_gold = self.model.translate(L2_sentences.to(self.device), 
                                                               language = 2,
                                                               gold = True)

                trans_err_L1 += language_similarity(L1_sentences_trans,
                                                    L1_sentences_trans_gold)
                trans_err_L2 += language_similarity(L2_sentences_trans,
                                                    L2_sentences_trans_gold)
                                                    
                L2_ms_from_gold = self.model.r(L1_sentences_trans_gold, 2) # M2 space from true translation
                                                                         # of L1 sentences. Expected to be aligned
                                                                         # with M1.

            L1_samples += L1_sentences.tolist()
            L2_samples += L2_sentences.tolist()

            L1_samples_trans += L1_sentences_trans.tolist()
            L2_samples_trans += L2_sentences_trans.tolist()

            untrained_M1_samples += untrained_L1_ms.detach().tolist()
            untrained_M2_samples += untrained_L2_ms.detach().tolist()

            gold_M1_samples += gold_L1_ms.detach().tolist()
            gold_M2_samples += gold_L2_ms.detach().tolist()

            M1_samples += L1_ms.detach().tolist()
            M2_samples += L2_ms.detach().tolist()

            M2_samples_from_gold += L2_ms_from_gold.detach().tolist()

            L1_samples_trans_gold += L1_sentences_trans_gold.detach().tolist()
            L2_samples_trans_gold += L2_sentences_trans_gold.detach().tolist()

            if self.configs['training']['adversial']:
                D_pred += torch.round(torch.sigmoid(dis_L1_logits)).detach().tolist()
                D_true += targets1.tolist()
                D_pred += torch.round(torch.sigmoid(dis_L2_logits)).detach().tolist()
                D_true += targets2.tolist()

        L1_samples = np.array(L1_samples)
        L2_samples = np.array(L2_samples)

        L1_samples_trans = np.array(L1_samples_trans)
        L2_samples_trans = np.array(L2_samples_trans)

        M1_samples = np.array(M1_samples)
        M2_samples = np.array(M2_samples)

        untrained_M1_samples = np.array(untrained_M1_samples)
        untrained_M2_samples = np.array(untrained_M2_samples)

        gold_M1_samples = np.array(gold_M1_samples)
        gold_M2_samples = np.array(gold_M2_samples)

        M2_samples_from_gold = np.array(M2_samples_from_gold)

        L1_samples_trans_gold = np.array(L1_samples_trans_gold)
        L2_samples_trans_gold = np.array(L2_samples_trans_gold)

        trans_err_L1 = trans_err_L1.item()
        trans_err_L2 = trans_err_L2.item()

        if self.configs['training']['adversial']:
            D_cm = confusion_matrix(D_true, D_pred)

        ############## STATS ###############

        L1_samples_mean = L1_samples.mean(axis = 0)
        L1_samples_std = L1_samples.std(axis = 0)
        L2_samples_mean = L2_samples.mean(axis = 0)
        L2_samples_std = L2_samples.std(axis = 0)
        M1_samples_mean = M1_samples.mean(axis = 0)
        M1_samples_std = M1_samples.std(axis = 0)
        M2_samples_mean = M2_samples.mean(axis = 0)
        M2_samples_std = M2_samples.std(axis = 0)

        # Ms overlap score
        M1_hull = M1_samples[ConvexHull(M1_samples).vertices]
        M2_hull = M2_samples[ConvexHull(M2_samples).vertices]

        M_samples = np.vstack([M1_samples, M2_samples])
        M_hull_union = M_samples[ConvexHull(M_samples).vertices]
        M_hull_union_area = PolyArea(M_hull_union[:,0],
                                     M_hull_union[:,1])

        intersect_res = intersect(M1_hull, 
                                  M2_hull)
        if len(intersect_res) > 0: # Check if it intersects
            M_hull_intersect = np.vstack(intersect_res)
        
            M_hull_intersect_area = PolyArea(M_hull_intersect[:,0],
                                            M_hull_intersect[:,1])
        else:
            M_hull_intersect_area = 0.

        Ms_overlap_score = 1. - (M_hull_union_area - M_hull_intersect_area)/M_hull_union_area

        # L1s overlap score
        L1_hull = L1_samples_trans[ConvexHull(L1_samples_trans).vertices]
        L1_gold_hull = L1_samples_trans_gold[ConvexHull(L1_samples_trans_gold).vertices]

        L1_concatenated_samples = np.vstack([L1_samples_trans, L1_samples_trans_gold])
        L1_hull_union = L1_concatenated_samples[ConvexHull(L1_concatenated_samples).vertices]
        L1_hull_union_area = PolyArea(L1_hull_union[:,0],
                                      L1_hull_union[:,1])

        intersect_res = intersect(L1_hull, 
                                  L1_gold_hull)
        if len(intersect_res) > 0: # Check if it intersects
            L1_hull_intersect = np.vstack(intersect_res)
        
            L1_hull_intersect_area = PolyArea(L1_hull_intersect[:,0],
                                              L1_hull_intersect[:,1])
        else:
            L1_hull_intersect_area = 0.

        L1s_overlap_score = 1. - (L1_hull_union_area - L1_hull_intersect_area)/L1_hull_union_area

        # L2s overlap score
        L2_hull = L2_samples_trans[ConvexHull(L2_samples_trans).vertices]
        L2_gold_hull = L2_samples_trans_gold[ConvexHull(L2_samples_trans_gold).vertices]

        L2_concatenated_samples = np.vstack([L2_samples_trans, L2_samples_trans_gold])
        L2_hull_union = L2_concatenated_samples[ConvexHull(L2_concatenated_samples).vertices]
        L2_hull_union_area = PolyArea(L2_hull_union[:,0],
                                      L2_hull_union[:,1])

        intersect_res = intersect(L2_hull, 
                                  L2_gold_hull)
        if len(intersect_res) > 0: # Check if it intersects
            L2_hull_intersect = np.vstack(intersect_res)
        
            L2_hull_intersect_area = PolyArea(L2_hull_intersect[:,0],
                                              L2_hull_intersect[:,1])
        else:
            L2_hull_intersect_area = 0.

        L2s_overlap_score = 1. - (L2_hull_union_area - L2_hull_intersect_area)/L2_hull_union_area

        # untrained Ms overlap score
        untrained_M1_hull = untrained_M1_samples[ConvexHull(
                                untrained_M1_samples
                                ).vertices]
        untrained_M2_hull = untrained_M2_samples[ConvexHull(
                                untrained_M2_samples
                                ).vertices]

        untrained_M_samples = np.vstack([untrained_M1_samples, 
                                         untrained_M2_samples])
        untrained_M_hull_union = untrained_M_samples[ConvexHull(
                                        untrained_M_samples
                                        ).vertices]
        untrained_M_hull_union_area = PolyArea(untrained_M_hull_union[:,0],
                                               untrained_M_hull_union[:,1])

        untrained_intersect_res = intersect(untrained_M1_hull, 
                                            untrained_M2_hull)
        if len(untrained_intersect_res) > 0:
            untrained_M_hull_intersect = np.vstack(untrained_intersect_res)
            
            untrained_M_hull_intersect_area = PolyArea(untrained_M_hull_intersect[:,0],
                                                    untrained_M_hull_intersect[:,1])
        else:
            untrained_M_hull_intersect_area = 0.

        untrained_Ms_overlap_score = 1. - (untrained_M_hull_union_area -\
                                           untrained_M_hull_intersect_area)/untrained_M_hull_union_area

        # Save score
        self._save_stats(L1_samples_mean,
                         L1_samples_std,
                         L2_samples_mean,
                         L2_samples_std,
                         M1_samples_mean,
                         M1_samples_std,
                         M2_samples_mean,
                         M2_samples_std,
                         Ms_overlap_score,
                         untrained_Ms_overlap_score,
                         L1s_overlap_score,
                         L2s_overlap_score,
                         trans_err_L1/n_batches,
                         trans_err_L2/n_batches)

        ############### PLOT ###############

        if plot or save:
            if metrics_historic or self.configs['training']['adversial']:
                fig, axs = plt.subplots(3, 3, figsize = (15,12))
                plt.subplots_adjust(wspace=0.35, hspace=0.3)
            else:
                fig, axs = plt.subplots(3, 2, figsize = (12,12))
            fig.suptitle('2D L and M spaces \n {} - prop. gold {} - {} \n {}{}'.format(
                                                            self.configs['name'],
                                                            self.configs['training']['prop_gold'],
                                                            'DAE' if self.configs['training']['denoising'] else '',
                                                            'shared enc.' if self.configs['training']['share_enc'] else '',
                                                            ' - shared dec.' if self.configs['training']['share_dec'] else ''
                                                            ),
                                                            fontsize=16)


            if plot:
                print("############# L STATS ###########")
                print("L1 mean : {} ; std : {}".format(np.round(
                                                            L1_samples_mean,
                                                            2),
                                                        np.round(
                                                            L1_samples_std,
                                                            2)
                                                        ))
                print("Trans. Err. {}".format(np.round(trans_err_L1/n_batches, 2)))
                print("L2 mean : {} ; std : {}".format(np.round(
                                                            L2_samples_mean,
                                                            2),
                                                        np.round(
                                                            L2_samples_std,
                                                            2)
                                                        ))
                print("Trans. Err. {}".format(np.round(trans_err_L2/n_batches, 2)))
                print("############# M STATS ###########")
                print("M1 mean : {} ; std : {}".format(np.round(
                                                            M1_samples_mean,
                                                            2),
                                                        np.round(
                                                            M1_samples_std,
                                                            2)
                                                        ))
                print("M2 mean : {} ; std : {}".format(np.round(
                                                            M2_samples_mean,
                                                            2),
                                                        np.round(
                                                            M2_samples_std,
                                                            2)
                                                        ))
                print("M's overlap score : {:.2f}".format(Ms_overlap_score))

            if self.configs['training']['adversial']:
                ### ADVERSIAL ###
                axs[2,2].set_title("Discriminator CM (Acc. {:.2f})".format(
                                                                    D_acc/n_batches
                                                                    ))
                axs[2,2] = sn.heatmap(D_cm, annot=True, fmt="d", cmap="YlGnBu")
                axs[2,2].set(xlabel='Prediction', ylabel='Truth')
                
                if not(metrics_historic):
                    # Disable remaining axes
                    axs[0,2].set_axis_off()
                    axs[1,2].set_axis_off()

            if metrics_historic:
                axs[0,2].set_title("Total Losses")
                if self.configs['training']['denoising']:
                    axs[0,2].plot(metrics_historic['DAE L1'],
                                 alpha = 0.5,
                                 marker = '+',
                                 label = 'DAE L1')
                    axs[0,2].plot(metrics_historic['DAE L2'],
                                 alpha = 0.5,
                                 marker = '+',
                                 label = 'DAE L2')

                try:
                    axs[0,2].plot(metrics_historic['BT L1'],
                                 alpha = 0.5,
                                 marker = '+',
                                 label = 'BT L1')
                    axs[0,2].plot(metrics_historic['BT L2'],
                                 alpha = 0.5,
                                 marker = '+',
                                 label = 'BT L2')
                except:
                    pass
                try:
                    axs[0,2].plot(metrics_historic['Trans L1'],
                                 alpha = 0.5,
                                 marker = '+',
                                 label = 'Trans Err L1')
                    axs[0,2].plot(metrics_historic['Trans L2'],
                                 alpha = 0.5,
                                 marker = '+',
                                 label = 'Trans Err L2')
                except:
                    pass
                axs[0,2].legend(loc='upper right')

                if self.configs['training']['adversial']:
                    axs[1,2].set_title("Discriminator")
                    axs[1,2].plot(metrics_historic['D Loss'],
                                  marker = '+',
                                  label = 'Loss')
                    axs[1,2].plot(metrics_historic['D Acc'],
                                  marker = '^',
                                  label = 'Acc')
                    axs[1,2].hlines(y=0.5,
                                    xmin = 0.,
                                    xmax = len(metrics_historic['D Loss'])-1,
                                    alpha = 0.5, 
                                    linestyle='--',
                                    linewidth=2, 
                                    color='g')
                    axs[1,2].hlines(y=1.,
                                    xmin = 0.,
                                    xmax = len(metrics_historic['D Loss'])-1,
                                    alpha = 0.5, 
                                    linestyle='--',
                                    linewidth=2, 
                                    color='r')
                    axs[1,2].legend(loc='upper right')
                else:
                    axs[1,2].set_axis_off()     
                    axs[2,2].set_axis_off()      
            

            ################ L #################
            L_x_min = min(
                        min(
                            L1_samples[:,0].min(),
                            L2_samples[:,0].min()
                        ),
                        min(
                            L1_samples_trans[:,0].min(),
                            L2_samples_trans[:,0].min()
                        )
                    )
            L_x_max = max(
                        max(
                            L1_samples[:,0].max(),
                            L2_samples[:,0].max()
                        ),
                        max(
                            L1_samples_trans[:,0].max(),
                            L2_samples_trans[:,0].max()
                        )
                    )
            L_y_min = min(
                        min(
                            L1_samples[:,1].min(),
                            L2_samples[:,1].min()
                        ),
                        min(
                            L1_samples_trans[:,1].min(),
                            L2_samples_trans[:,1].min()
                        )
                    )
            L_y_max = max(
                        max(
                            L1_samples[:,1].max(),
                            L2_samples[:,1].max()
                        ),
                        max(
                            L1_samples_trans[:,1].max(),
                            L2_samples_trans[:,1].max()
                        )
                    )

            # Ls

            axs[0,0].scatter(L1_samples[:,0],
                            L1_samples[:,1], 
                            alpha = 0.5,
                            label = 'L1')
            axs[0,0].scatter(L2_samples[:,0],
                            L2_samples[:,1], 
                            alpha = 0.5,
                            label = 'L2')


            axs[0,0].set_title("L space")
            axs[0,0].legend()
            axs[0,0].grid(True)

            axs[0,0].set_xlim([L_x_min, L_x_max])
            axs[0,0].set_ylim([L_y_min, L_y_max])

            # Translated Ls

            axs[0,1].scatter(L2_samples_trans[:,0],
                            L2_samples_trans[:,1], 
                            alpha = 0.5,
                            label = 'L1 (from L2)')
            axs[0,1].scatter(L1_samples_trans[:,0],
                            L1_samples_trans[:,1], 
                            alpha = 0.5,
                            label = 'L2 (from L1)')


            axs[0,1].set_title("Translated L space")
            axs[0,1].legend()
            axs[0,1].grid(True)
            
            axs[0,1].set_xlim([L_x_min, L_x_max])
            axs[0,1].set_ylim([L_y_min, L_y_max])

            # Arrows
            
            Lx_trans_effect = L1_samples_trans_gold[:,0] - L1_samples_trans[:,0]
            Ly_trans_effect = L1_samples_trans_gold[:,1] - L1_samples_trans[:,1]
            
            color_array = np.sqrt(Lx_trans_effect**2 + Ly_trans_effect**2)

            qq = axs[1,1].quiver(L1_samples_trans[:,0],
                                L1_samples_trans[:,1], 
                                Lx_trans_effect,
                                Ly_trans_effect,
                                color_array)
            qq_pos = axs[1,1].get_position()
            cax = fig.add_axes([qq_pos.x0 + 1.05*qq_pos.width, 
                                qq_pos.y0,
                                qq_pos.width / 15,
                                qq_pos.height])
            plt.colorbar(qq, cax = cax)
            
            axs[1,1].scatter(L1_samples_trans[:,0],
                            L1_samples_trans[:,1], 
                            alpha = 0.3,
                            c = 'tab:orange',
                            label = 'L2 (from L1)')
            axs[1,1].scatter(L1_samples_trans_gold[:,0],
                            L1_samples_trans_gold[:,1], 
                            alpha = 0.15,
                            c = 'orange',
                            label = 'L2 (from L1 gold)')

            axs[1,1].set_title("L spaces alignment")
            axs[1,1].legend()
            axs[1,1].set_aspect('equal')
            axs[1,1].grid(True)
            
            ################## M ##################
            M_x_min = min(
                        min(
                            gold_M1_samples[:,0].min(),
                            gold_M2_samples[:,0].min()
                            ),
                        min(
                            min(
                                M1_samples[:,0].min(),
                                M2_samples[:,0].min()
                            ),
                            min(
                                untrained_M1_samples[:,0].min(),
                                untrained_M2_samples[:,0].min()
                            )
                        )
                    )
            M_x_max = max( 
                        max(
                            gold_M1_samples[:,0].max(),
                            gold_M2_samples[:,0].max()
                        ),
                        max(
                            max(
                                M1_samples[:,0].max(),
                                M2_samples[:,0].max()
                            ),
                            max(
                                untrained_M1_samples[:,0].max(),
                                untrained_M2_samples[:,0].max()
                            )
                        )
                    )
            M_y_min = min(
                        min(
                            gold_M1_samples[:,1].min(),
                            gold_M2_samples[:,1].min()
                        ),
                        min(
                            min(
                                M1_samples[:,1].min(),
                                M2_samples[:,1].min()
                            ),
                            min(
                                untrained_M1_samples[:,1].min(),
                                untrained_M2_samples[:,1].min()
                            )
                        )
                    )
                    
            M_y_max = max(
                        max(
                            gold_M1_samples[:,1].max(),
                            gold_M2_samples[:,1].max()
                            ),
                        max(
                            max(
                                M1_samples[:,1].max(),
                                M2_samples[:,1].max()
                            ),
                            max(
                                untrained_M1_samples[:,1].max(),
                                untrained_M2_samples[:,1].max()
                            )
                        )
                    )   
            
            # Untrained

            axs[1,0].scatter(untrained_M1_samples[:,0],
                             untrained_M1_samples[:,1], 
                             alpha = 0.5,
                             marker= "s",
                             label = 'M1 (untrained)')
            axs[1,0].scatter(untrained_M2_samples[:,0],
                             untrained_M2_samples[:,1], 
                             alpha = 0.5,
                             marker = "s",
                             label = 'M2 (untrained)')
            axs[1,0].scatter(gold_M1_samples[:,0],
                             gold_M1_samples[:,1], 
                             alpha = 0.5,
                             marker= "s",
                             label = 'M1 (gold)')
            axs[1,0].scatter(gold_M2_samples[:,0],
                             gold_M2_samples[:,1], 
                             alpha = 0.5,
                             marker = "s",
                             label = 'M2 (gold)')


            axs[1,0].set_title("M space - before training")
            axs[1,0].legend()
            axs[1,0].grid(True)

            axs[1,0].set_xlim([M_x_min, M_x_max])
            axs[1,0].set_ylim([M_y_min, M_y_max])

            # Trained


            axs[2,0].scatter(M1_samples[:,0],
                            M1_samples[:,1], 
                            alpha = 0.5,
                            marker= "D",
                            label = 'M1')
            axs[2,0].scatter(M2_samples[:,0],
                            M2_samples[:,1], 
                            alpha = 0.5,
                            marker = "D",
                            label = 'M2')


            axs[2,0].set_title("M space - after training")
            axs[2,0].legend()
            axs[2,0].grid(True)

            axs[2,0].set_xlim([M_x_min, M_x_max])
            axs[2,0].set_ylim([M_y_min, M_y_max])

            # Arrows
            Mx_trans_effect = M2_samples_from_gold[:,0] - M1_samples[:,0]
            My_trans_effect = M2_samples_from_gold[:,1] - M1_samples[:,1]
            
            color_array = np.sqrt(Mx_trans_effect**2 + My_trans_effect**2)

            qq = axs[2,1].quiver(M1_samples[:,0],
                                M1_samples[:,1], 
                                Mx_trans_effect,
                                My_trans_effect,
                                color_array)#,
                                #angles = 'xy',
                                #scale = 1.,
                                #scale_units = 'xy')
            qq_pos = axs[2,1].get_position()
            cax = fig.add_axes([qq_pos.x0 + 1.05*qq_pos.width, 
                                qq_pos.y0,
                                qq_pos.width / 15,
                                qq_pos.height])
            plt.colorbar(qq, cax = cax)#, cmap = plt.cm.jet)

            axs[2,1].scatter(M1_samples[:,0],
                            M1_samples[:,1], 
                            alpha = 0.2,
                            marker= "D",
                            label = 'M1')
            axs[2,1].scatter(M2_samples_from_gold[:,0],
                            M2_samples_from_gold[:,1], 
                            alpha = 0.2,
                            marker = "D",
                            label = 'M2')

            axs[2,1].set_title("M spaces alignment")
            axs[2,1].grid(True)
            axs[2,1].set_aspect('equal')


            if plot:
                plt.show()
            if save:
                plt.savefig(
                    os.path.join(
                        self.saving_root,
                        f"{self.imgs_saved:03}" + '.png' 
                        )
                )
                self.imgs_saved += 1
            plt.close()

    def _save_stats(self, L1_mean: np.ndarray,
                          L1_std: np.ndarray,
                          L2_mean: np.ndarray,
                          L2_std: np.ndarray,
                          M1_mean: np.ndarray,
                          M1_std: np.ndarray,
                          M2_mean: np.ndarray,
                          M2_std: np.ndarray,
                          Ms_overlap_score: float,
                          untrained_Ms_overlap_score: float,
                          L1s_overlap_score: float,
                          L2s_overlap_score: float,
                          trans_err_L1: float,
                          trans_err_L2: float) -> None:
        """
            Helper fucntion.
            Save everything in a dict.
        """

        self.stats['L1 mean'] = L1_mean
        self.stats['L1 std'] = L1_std
        self.stats['L2 mean'] = L2_mean
        self.stats['L2 std'] = L2_std
        self.stats['M1 mean'] = M1_mean
        self.stats['M1 std'] = M1_std
        self.stats['M2 mean'] = M2_mean
        self.stats['M2 std'] = M2_std
        self.stats['Ms overlap score'] = Ms_overlap_score
        self.stats['untrained Ms overlap score'] = untrained_Ms_overlap_score
        self.stats['L1s overlap score'] = L1s_overlap_score
        self.stats['L2s overlap score'] = L2s_overlap_score
        self.stats['trans err L1'] = trans_err_L1
        self.stats['trans err L2'] = trans_err_L2