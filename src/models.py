"""
    blabla
"""   
import os
from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.helpers import freeze_module, get_gold_params_continuous_squares, get_gold_params_gaussian_mixture, get_init_state, get_mask
from src.modules import LayerNorm, MultiheadAttention, SinusoidalPositionalEmbedding

class BackTranslator(nn.Module):
    """
        This class is a wrapper for back translation in a sense
        that it is highly modular, the model can :
            - use shared encoder or decoder
            - use different encoder or decoder
            - do Denoising Auto Encoding (DAE)
            - do Back Translation (BT)
            - more to come !

        DAE : x_1:L1 -r1-> M1 – s1 generates -> x_1
              x_2:L2 -r2-> M2 – s2 generates -> x_2

        BT : x_1:L1 -r1-> M1 -> T_M^{12} -> M2 – s2 generates -> y_2 -r2-> M2 -> T_M^{21} -> M1 - s1 predicts -> x_1
             x_2:L2 -r2-> M2 -> T_M^{21} -> M1 – s1 generates -> y_1 -r1-> M1 -> T_M^{12} -> M2 - s2 predicts -> x_2
             for now T_M^{12} = T_M^{21} = Id 
    """
    def __init__(self, configs: dict,
                       device: str,
                       stochastic: bool = True,
                       train_all: bool = False # backpropagate trought the whole BT pipeline
                                               # requires Gumbel-Softmax
                       ) -> None:
        super(BackTranslator, self).__init__()
        
        # Needed in forward pass
        self.train_all = train_all
        self.device = device
        # Generate via sampling
        self.stochastic = stochastic
        
        # Either ways we train those
        if configs['models']['encoder']['model'] == 'mlp':
            self.r = MLPEnc(configs)
        elif configs['models']['encoder']['model'] == 'lstm':
            self.r = LSTMEnc(configs)
        elif configs['models']['encoder']['model'] == 'transformer':
            self.r = TransformerEnc(configs)

        if configs['models']['decoder']['model'] == 'mlp':
            self.s = MLPDec(configs)
        elif configs['models']['decoder']['model'] == 'lstm':
            self.s = LSTMDec(configs, self.r)
        elif configs['models']['decoder']['model'] == 'transformer':
            self.s = TransformerDec(configs, self.r)

        # Gold 
        if not(configs['dataset']['only_gold_translation']):
            self.gold_enc = GoldEncoder(configs)
            self.gold_dec = GoldDecoder(configs)
        
    def denoising_autoencoding(self, inputs: dict, 
                                     language: int) -> torch.Tensor:
        """
            Forward pass of the DAE objective.
        """
        # Encoding
        m = self.r(inputs, language)
        # Needed to decode in LSTM
        m['y'] = inputs['x']
        # Decoding
        recons = self.s(m, language)
        return recons

    def back_translation(self, inputs: dict,
                               language: int,
                               gold: bool = False) -> torch.Tensor:
        """
            Forward pass of the BT objective.
            If gold is set to True then the training is supervised.
        """
        if language == 1:
            lang1, lang2 = 1, 2
        elif language == 2:
            lang1, lang2 = 2, 1
        else:
            raise Exception("language should be an int equals to 1 or 2.")

        if self.train_all:
            # Not working now in stochastic setup because 
            # of the generation (cannot backpropagate)
            m = self.r(inputs, lang1)
            if self.stochastic:
                # Need Gumbel-Softmax
                sentences_prime_probs = F.softmax(self.s(m, lang2), dim = -1)
            else:
                sentences_prime = self.s(m, lang2)
        else:
            with torch.no_grad():
                if gold:
                    if 'gold translation' in inputs.keys():
                        ### Create the forward pass inputs dict ###
                        inputs_prime = {'x':  inputs['gold translation'],
                                        'lengths':  inputs['gold lengths']}
                    else:
                        m = self.gold_enc(inputs, lang1)
                        sentences_prime = self.gold_dec(m, lang2)
                        ### Create the forward pass inputs dict ###
                        inputs_prime = {'x': sentences_prime}
                else:
                    m = self.r(inputs, lang1)
                    if self.stochastic:
                        # Compute probability distribution
                        sentences_prime, lengths_prime, _ = self.s.generate(m, lang2)
                        ### Create the forward pass inputs dict ###
                        inputs_prime = {'x': sentences_prime,
                                        'lengths': lengths_prime}
                    else:
                        sentences_prime = self.s(m, lang2)
                        ### Create the forward pass inputs dict ###
                        inputs_prime = {'x': sentences_prime}
        
        m_prime = self.r(inputs_prime, lang2)
        # Needed to decode in LSTM
        m_prime['y'] = inputs['x']
        # Decoding
        sentences_recons = self.s(m_prime, lang1)
        
        return sentences_recons

    def translate(self, inputs: dict,
                        language: int,
                        gold: bool = False,
                        output_scores: bool = False) -> torch.Tensor:
        """
            Translate sentences (from language)
            according to :
                - T_L^{12} = s2 o T_M_{12} o r1
                - T_L^{21} = s1 o T_M_{21} o r2
        """

        # In the stochastic scenario translation is a 
        # probability distribution.
        # Do we take an argmax here ?
        assert not(gold and output_scores)

        if language == 1:
            lang1, lang2 = 1, 2
        elif language == 2:
            lang1, lang2 = 2, 1
        else:
            raise Exception("language should be an int equals to 1 or 2.")
        with torch.no_grad():
            if gold:
                if 'gold tanslation' in inputs.keys():
                    return inputs['gold translation']
                else:
                    m_gold = self.gold_enc(inputs,
                                        language)
                    gold_translation = self.gold_dec(m_gold,
                                                    lang2) # = 1 if language == 2
                                                            # = 2 if language == 1
                return gold_translation
            else:
                scores = None
                m = self.r(inputs, lang1)
                
                if self.stochastic:
                    translation, _, _ = self.s.generate(m, lang2)
                    if output_scores:
                        m['y'] = inputs['gold translation']
                        scores = self.s(m, lang2)
                else:
                    translation = self.s(m, lang2)
                    if output_scores:
                        scores = translation
                return translation, scores

############################################################################
##                                                                        ##
##                        GOLD ENCODERS/DECODERS                          ##
##                                                                        ##
############################################################################     


class GoldEncoder(nn.Module):
    """
        R_i : L_i -> M_i
        language to meaning (gold label)
    """
    def __init__(self, configs: dict) -> None:
        super(GoldEncoder, self).__init__()
        
        if configs['name'] in ['ContinuousSquares', 'GaussianMixture']:
            if configs['name'] == 'ContinuousSquares':
                enc_L1_weights, enc_L1_bias, enc_L2_weights, enc_L2_bias, _, _, _, _ = get_gold_params_continuous_squares(configs)
            elif configs['name'] == 'GaussianMixture':
                enc_L1_weights, enc_L1_bias, enc_L2_weights, enc_L2_bias, _, _, _, _ = get_gold_params_gaussian_mixture(configs)
            # L1
            self.enc_L1 = nn.Linear(configs['dataset']['L']['dim'], 
                                    configs['dataset']['M']['dim'], 
                                    bias = True)
            self.enc_L1.weight = torch.nn.Parameter(
                                        enc_L1_weights)
            self.enc_L1.bias = torch.nn.Parameter(
                                        enc_L1_bias)
            # L2
            self.enc_L2 = nn.Linear(configs['dataset']['L']['dim'], 
                                    configs['dataset']['M']['dim'], 
                                    bias = True)
            self.enc_L2.weight = torch.nn.Parameter(
                                    enc_L2_weights)
            self.enc_L2.bias = torch.nn.Parameter(
                                    enc_L2_bias)  # Ensure that R2oS2 = Id
                                                  # We do not train the gold label
            freeze_module(self.enc_L1) 
            freeze_module(self.enc_L2)
        else:
            raise Exception("No Gold Encoder for {}!".format(configs['name']))
        
    def forward(self, inputs: dict, 
                      language: int) -> dict:
        """
            Forward pass of the GOLD encoder.
        """
        x = inputs['x']
        if language == 1:
            m = self.enc_L1(x)
            return {'m': m}
        elif language == 2:
            m =  self.enc_L2(x)
            return {'m': m}
        else:
            raise Exception("language should be an int equals to 1 or 2.")


class GoldDecoder(nn.Module):
    """
        S_i : M_i -> L_i
        meaning to language (gold label)
    """
    def __init__(self, configs: dict) -> None:
        super(GoldDecoder, self).__init__()
        
        if configs['name'] in ['ContinuousSquares', 'GaussianMixture']:
            if configs['name'] == 'ContinuousSquares':
                _, _, _, _, dec_L1_weights, dec_L1_bias, dec_L2_weights, dec_L2_bias = get_gold_params_continuous_squares(configs)
            elif configs['name'] == 'GaussianMixture':
                _, _, _, _, dec_L1_weights, dec_L1_bias, dec_L2_weights, dec_L2_bias = get_gold_params_gaussian_mixture(configs)
            # L1
            self.dec_L1 = nn.Linear(configs['dataset']['M']['dim'], 
                                    configs['dataset']['L']['dim'], 
                                    bias = True)
            self.dec_L1.weight = torch.nn.Parameter(
                                        dec_L1_weights)
            self.dec_L1.bias = torch.nn.Parameter(
                                        dec_L1_bias
                                        )
            # L2
            self.dec_L2 = nn.Linear(configs['dataset']['M']['dim'], 
                                    configs['dataset']['L']['dim'], 
                                    bias = True)
            self.dec_L2.weight = torch.nn.Parameter(
                                        dec_L2_weights)
            self.dec_L2.bias = torch.nn.Parameter(
                                        dec_L2_bias
                                        ) 
            freeze_module(self.dec_L1) 
            freeze_module(self.dec_L2)
        else:
            raise Exception("No Gold Encoder for {}!".format(configs['name']))
        
    def forward(self, inputs: dict, language: int) -> torch.Tensor:
        """
            Forward pass of the GOLD decoder.
        """
        m = inputs['m']
        if language == 1:
            l = self.dec_L1(m)
            return l
        elif language == 2:
            l =  self.dec_L2(m)
            return l
        else:
            raise Exception("language should be an int equals to 1 or 2.")


#########################################################################################
# Discriminator

class Discriminator(nn.Module):
    """
        Discriminator that tries to predict if m 
        belongs to M1 or M2. Intuitively it should
        enforce M1 and M2 to be superposed.
    """
    def __init__(self, configs: dict) -> None:
        super(Discriminator, self).__init__()
        
        if configs['model'] == 'linear':
            self.dis = nn.Linear(configs['linear']['dim_in'], 
                                 configs['linear']['dim_out'])
        elif configs['model'] == 'mlp':
            self.dis = nn.Sequential(
                        nn.Linear(
                            configs['mlp']['dim_in'], 
                            configs['mlp']['dim_hidden']
                            ),
                        nn.LeakyReLU(),
                        nn.Linear(
                            configs['mlp']['dim_hidden'], 
                            configs['mlp']['dim_out']
                            )
                        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.dis(input)
        return logits.squeeze()


#######################################################################################
#                                                                                      #
#                                        MLP                                           #
#                                                                                      #
########################################################################################

class MLPEnc(nn.Module):

    def __init__(self, configs):
        super(MLPEnc, self).__init__()

        configs_enc = configs['models']['encoder']['mlp']
        # MLPs layers / shared layers
        if configs['training']['share_enc']:
            mlp = nn.Sequential(
                            nn.Linear(
                                configs_enc['dim_in'], 
                                configs_enc['dim_hidden']
                                ),
                            nn.LeakyReLU(),
                            nn.Linear(
                                configs_enc['dim_hidden'], 
                                configs_enc['dim_out']
                                )
                            )
            mlps = [mlp, mlp]
        else:
            mlps = [
                    nn.Sequential(
                            nn.Linear(
                                configs_enc['dim_in'], 
                                configs_enc['dim_hidden']
                                ),
                            nn.LeakyReLU(),
                            nn.Linear(
                                configs_enc['dim_hidden'], 
                                configs_enc['dim_out']
                                )
                            )
                    for _ in range(2)
                    ]
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs: dict, 
                      language: int):
        x = inputs['x']
        enc = self.mlps[language-1]
        return {'m': enc(x)}

class MLPDec(nn.Module):

    def __init__(self, configs):
        super(MLPDec, self).__init__()

        configs_dec = configs['models']['decoder']['mlp']
        # MLPs layers / shared layers
        if configs['training']['share_dec']:
            mlp = nn.Sequential(
                            nn.Linear(
                                configs_dec['dim_in'], 
                                configs_dec['dim_hidden']
                                ),
                            nn.LeakyReLU(),
                            nn.Linear(
                                configs_dec['dim_hidden'], 
                                configs_dec['dim_out']
                                )
                            )
            mlps = [mlp, mlp]
        else:
            mlps = [
                    nn.Sequential(
                            nn.Linear(
                                configs_dec['dim_in'], 
                                configs_dec['dim_hidden']
                                ),
                            nn.LeakyReLU(),
                            nn.Linear(
                                configs_dec['dim_hidden'], 
                                configs_dec['dim_out']
                                )
                            )
                    for _ in range(2)
                    ]
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs, language):
        m = inputs['m']
        dec = self.mlps[language-1]
        return dec(m)


########################################################################################
#                                                                                      #
#                                   Seq2Seq                                            #
#                                                                                      #
########################################################################################
# From Lample et al. https://github.com/facebookresearch/UnsupervisedMT

LatentState = namedtuple('LatentState', 'dec_input, dis_input, input_len')
LSTM_PARAMS = ['weight_ih_l%i', 'weight_hh_l%i', 'bias_ih_l%i', 'bias_hh_l%i']

class LSTMEnc(nn.Module):

    ENC_ATTR = ['n_langs', 
                'n_words', 
                ('share_lang_emb', False), 
                'emb_dim', 
                'hidden_dim', 
                'dropout', 
                'n_enc_layers', 
                'enc_dim', 
                ('share_enc', False), 
                'proj_mode', 
                'pad_index']

    def __init__(self, configs):
        """
        Encoder initialization.
        """
        super(LSTMEnc, self).__init__()
        # Configs for encoder
        configs_enc = configs['models']['encoder']['lstm']
        # language parameters
        self.n_langs = 2
        self.n_words = [configs['dataset']['vocab_size'] for _ in range(2)] # /!\ Only work if vocabulary is shared /!\
        self.pad_index = configs['dataset']['pad_index']
        # model parameters
        self.share_lang_emb = configs_enc['share_lang_emb']
        self.emb_dim = configs_enc['emb_dim']
        self.hidden_dim = configs_enc['hidden_dim']
        self.dropout = configs_enc['dropout']
        self.n_enc_layers = configs_enc['n_enc_layers']
        self.enc_dim = configs_enc['enc_dim']
        self.share_enc = configs['training']['share_enc'] # /!\ to change /!\
        self.proj_mode = configs_enc['proj_mode']
        self.freeze_enc_emb = configs_enc['freeze_enc_emb']
        assert not self.share_lang_emb or len(set(self.n_words)) == 1
        assert 0 <= self.share_enc <= self.n_enc_layers + int(self.proj_mode == 'proj')

        # embedding layers
        if self.share_lang_emb: # In practice True
            print("Sharing encoder input embeddings")
            layer_0 = nn.Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
            # Loading pre-computed embeddings weights
            self.embed_layer_name = "data\\embeddings\\simple-{}-{}_voc_{}.pt".format(
                                                    configs['dataset']['L1']['name'], 
                                                    configs['dataset']['L2']['name'],
                                                    configs['dataset']['vocab_size'])
            if os.path.exists(self.embed_layer_name):
                print('Loading pre-computed embeddings weights...')
                layer_0.load_state_dict(torch.load(
                    self.embed_layer_name
                ))
            else:
                nn.init.normal_(layer_0.weight, 0, 0.1)
                nn.init.constant_(layer_0.weight[self.pad_index], 0)

            embeddings = [layer_0 for _ in range(self.n_langs)]
        else:
            embeddings = []
            for n_words in self.n_words:
                layer_i = nn.Embedding(n_words, self.emb_dim, padding_idx=self.pad_index)
                nn.init.normal_(layer_i.weight, 0, 0.1)
                nn.init.constant_(layer_i.weight[self.pad_index], 0)

                embeddings.append(layer_i)
        self.embeddings = nn.ModuleList(embeddings)

        # LSTM layers / shared layers
        lstm = [
                nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=self.n_enc_layers, dropout=self.dropout)
                for _ in range(self.n_langs)
                ]
        for k in range(self.n_enc_layers):
            if self.n_enc_layers - k <= self.share_enc - int(self.proj_mode == 'proj'):
                print("Sharing encoder LSTM parameters for layer %i" % k)
                for i in range(1, self.n_langs):
                    for name in LSTM_PARAMS:
                        setattr(lstm[i], name % k, getattr(lstm[0], name % k))
        self.lstm = nn.ModuleList(lstm)

        # projection layers
        if self.proj_mode == 'proj':
            if self.share_enc >= 1:
                print("Sharing encoder projection layers")
                proj_0 = nn.Linear(self.hidden_dim, self.enc_dim)
                proj = [proj_0 for _ in range(self.n_langs)]
            else:
                proj = [nn.Linear(self.hidden_dim, self.enc_dim) for _ in range(self.n_langs)]
            self.proj = nn.ModuleList(proj)
        else:
            self.proj = [None for _ in range(self.n_langs)]

    def forward(self, inputs, language):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
            - LongTensor of size (bs,), sentence lengths
        Output:
            - FloatTensor of size (bs, enc_dim),
              representing the encoded state of each sentence
        """
        x = torch.transpose(inputs['x'], 0, 1)
        #print('x ', x)
        lengths = inputs['lengths']
        lang_id = language - 1 #/!\ Let say lang_id = language - 1 
        assert type(lang_id) is int
        is_cuda = x.is_cuda
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        lstm_layer.flatten_parameters() # To force the weights to be contiguous in memory
        proj_layer = self.proj[lang_id]

        # embeddings
        slen, bs = x.size(0), x.size(1)
        if x.dim() == 2:
            embeddings = emb_layer(x)
        else: # In case the embedding is not shared (looks hard)
            assert x.dim() == 3 and x.size(2) == self.n_words[lang_id]
            embeddings = x.view(slen * bs, -1).mm(emb_layer.weight).view(slen, bs, self.emb_dim)
        
        embeddings = embeddings.detach() if self.freeze_enc_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training) 

        #assert lengths.max() == slen and lengths.size(0) == bs # False for us
        assert embeddings.size() == (slen, bs, self.emb_dim)
        #print("embeddings ", embeddings)

        # LSTM

        #######################################################
        embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, 
                                                             lengths, 
                                                             enforce_sorted=False,
                                                             batch_first=False)
        #######################################################
        lstm_output, (_, _) = lstm_layer(embeddings)

        #######################################################
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output,
                                                          total_length = slen, 
                                                          batch_first=False)

        #######################################################
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # encoded sentences representation
        if self.proj_mode == 'pool':
            latent_state = lstm_output.max(0)[0]
        else:
            # select the last state of each sentence
            mask = get_mask(lengths, 
                            slen, 
                            False, 
                            expand=self.hidden_dim, 
                            batch_first=True, 
                            cuda=is_cuda)
            h_t = lstm_output.transpose(0, 1).masked_select(mask).view(bs, self.hidden_dim)
            #print("h_t ", h_t)
            if self.proj_mode == 'proj':
                latent_state = proj_layer(h_t)
            elif self.proj_mode == 'last':
                latent_state = h_t

        #print("latent_state shape ", latent_state.shape)
        #print("latent_state ", latent_state)
        #print("lengths ", lengths)
        #exit(0)

        return {'m': LatentState(input_len=lengths, 
                                 dec_input=latent_state, 
                                 dis_input=latent_state)}


class LSTMDec(nn.Module):

    DEC_ATTR = ['n_langs', 'n_words', 
                ('share_lang_emb', False), 
                ('share_encdec_emb', False), 
                ('share_decpro_emb', False), 
                ('share_dec', False), 
                'emb_dim', 
                'hidden_dim', 
                'dropout', 
                'n_dec_layers', 
                'enc_dim', 
                'init_encoded', 
                'eos_index', 
                'pad_index', 
                'bos_index']

    def __init__(self, configs, encoder):
        """
        Decoder initialization.
        """
        super(LSTMDec, self).__init__()
        # Configs for encoder
        configs_dec = configs['models']['decoder']['lstm']
        # language parameters
        self.n_langs = 2
        self.max_len = configs['dataset']['max_len']
        self.n_words = [configs['dataset']['vocab_size'] for _ in range(2)] # /!\ Only work if vocabulary is shared /!\
        self.eos_index = configs['dataset']['eos_index'] # End of the sentence
        self.pad_index = configs['dataset']['pad_index']
        self.bos_index = [configs['dataset']['bos{}_index'.format(k+1)] for k in range(2)] # Beginning /!\ MUST NO BE THE SAME /!\
        # model parameters
        self.share_lang_emb = configs_dec['share_lang_emb']
        self.share_encdec_emb = configs_dec['share_encdec_emb']
        self.share_decpro_emb = configs_dec['share_decpro_emb']
        self.share_output_emb = configs_dec['share_output_emb']
        self.share_lstm_proj = configs_dec['share_lstm_proj']
        self.share_dec = configs['training']['share_dec'] # /!\ To change /!\
        self.emb_dim = configs_dec['emb_dim']
        self.hidden_dim = configs_dec['hidden_dim']
        self.lstm_proj = configs_dec['lstm_proj']
        self.dropout = configs_dec['dropout']
        self.n_dec_layers = configs_dec['n_dec_layers']
        self.enc_dim = configs_dec['enc_dim']
        self.init_encoded = configs_dec['init_encoded']
        self.freeze_dec_emb = configs_dec['freeze_dec_emb']
        assert not self.share_lang_emb or len(set(self.n_words)) == 1
        assert not self.share_decpro_emb or self.lstm_proj or self.emb_dim == self.hidden_dim
        assert 0 <= self.share_dec <= self.n_dec_layers
        assert self.enc_dim == self.hidden_dim or not self.init_encoded
        

        # words allowed for generation
        #self.vocab_mask_neg = params.vocab_mask_neg if len(params.vocab) > 0 else None
        self.vocab_mask_neg = None

        # embedding layers
        if self.share_encdec_emb: # /!\ In practice True
            print("Sharing encoder and decoder input embeddings")
            embeddings = encoder.embeddings
        else:
            if self.share_lang_emb:
                print("Sharing decoder input embeddings")
                layer_0 = nn.Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
                nn.init.normal_(layer_0.weight, 0, 0.1)
                nn.init.constant_(layer_0.weight[self.pad_index], 0)

                embeddings = [layer_0 for _ in range(self.n_langs)]
            else:
                embeddings = []
                for n_words in self.n_words:
                    layer_i = nn.Embedding(n_words, self.emb_dim, padding_idx=self.pad_index)
                    nn.init.normal_(layer_i.weight, 0, 0.1)
                    nn.init.constant_(layer_i.weight[self.pad_index], 0)

                    embeddings.append(layer_i)
            embeddings = nn.ModuleList(embeddings)
        self.embeddings = embeddings

        # LSTM layers / shared layers
        input_dim = self.emb_dim + (0 if self.init_encoded else self.enc_dim)
        lstm = [
            nn.LSTM(input_dim, self.hidden_dim, num_layers=self.n_dec_layers, dropout=self.dropout)
            for _ in range(self.n_langs)
        ]
        for k in range(self.n_dec_layers):
            if k + 1 <= self.share_dec:
                print("Sharing decoder LSTM parameters for layer %i" % k)
                for i in range(1, self.n_langs):
                    for name in LSTM_PARAMS:
                        setattr(lstm[i], name % k, getattr(lstm[0], name % k))
        self.lstm = nn.ModuleList(lstm)

        # projection layers between LSTM and output embeddings
        if self.lstm_proj:
            lstm_proj_layers = [nn.Linear(self.hidden_dim, self.emb_dim) for _ in range(self.n_langs)]
            if self.share_lstm_proj:
                print("Sharing decoder post-LSTM projection layers")
                for i in range(1, self.n_langs):
                    lstm_proj_layers[i].weight = lstm_proj_layers[0].weight
                    lstm_proj_layers[i].bias = lstm_proj_layers[0].bias
            self.lstm_proj_layers = nn.ModuleList(lstm_proj_layers)
            proj_output_dim = self.emb_dim
        else:
            self.lstm_proj_layers = [None for _ in range(self.n_langs)]
            proj_output_dim = self.hidden_dim

        # projection layers
        proj = [nn.Linear(proj_output_dim, n_words) for n_words in self.n_words] # Maybe should not count [PAD] as a possibility
        if self.share_decpro_emb:
            print("Sharing input embeddings and projection matrix in the decoder")
            for i in range(self.n_langs):
                proj[i].weight = self.embeddings[i].weight
            if self.share_lang_emb:
                assert self.share_output_emb
                print("Sharing decoder projection matrices")
                for i in range(1, self.n_langs):
                    proj[i].bias = proj[0].bias
        elif self.share_output_emb:
            assert self.share_lang_emb
            print("Sharing decoder projection matrices")
            for i in range(1, self.n_langs):
                proj[i].weight = proj[0].weight
                proj[i].bias = proj[0].bias
        self.proj = nn.ModuleList(proj)

    def forward(self, inputs, language, one_hot=False):
        """
        Input:
            - LongTensor of size (slen, bs), word indices
              or
              LongTensor of size (slen, bs, n_words), one-hot word embeddings
            - LongTensor of size (bs,), sentence lengths
            - FloatTensor of size (bs, hidden_dim), latent
              state representing sentences
        Output:
            - FloatTensor of size (slen, bs, n_words),
              representing the score of each word in each sentence
        """
        encoded = inputs['m']
        y = torch.transpose(inputs['y'], 0, 1)[:-1] # To avoid feeding BOS in loss
        lang_id = language - 1
        assert type(lang_id) is int
        assert encoded.input_len.size(0) == encoded.dec_input.size(0)
        latent = encoded.dec_input
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        lstm_layer.flatten_parameters() # To force the weights to be contiguous in memory
        lstm_proj_layer = self.lstm_proj_layers[lang_id]
        proj_layer = self.proj[lang_id]

        # embeddings
        if one_hot:
            slen, bs, _ = y.size()
            embeddings = y.view(slen * bs, n_words).mm(emb_layer.weight)
            embeddings = embeddings.view(slen, bs, self.emb_dim)
        else:
            slen, bs = y.size()
            embeddings = emb_layer(y)
        embeddings = embeddings.detach() if self.freeze_dec_emb else embeddings
        embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        assert latent.size() == (bs, self.enc_dim)
        assert embeddings.size() == (slen, bs, self.emb_dim)

        if self.init_encoded:
            init = get_init_state(self.n_dec_layers, bs, self.hidden_dim, latent)
            lstm_input = embeddings
        else:
            init = None
            encoded = latent.unsqueeze(0).expand(slen, bs, self.enc_dim)
            lstm_input = torch.cat([embeddings, encoded], 2)

        # LSTM
        lstm_output, (_, _) = lstm_layer(lstm_input, init)
        assert lstm_output.size() == (slen, bs, self.hidden_dim)

        # word scores
        output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(-1, self.hidden_dim)
        if lstm_proj_layer is not None:
            output = F.relu(lstm_proj_layer(output))
        scores = proj_layer(output)
        return torch.transpose(
                    torch.transpose(
                        scores.view(slen, bs, n_words), 
                        0, 1), 
                    1, 2)

    def generate(self, inputs, language, sample=True, temperature=1.):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence lengths
        """
        encoded = inputs['m']
        lang_id = language - 1
        assert encoded.input_len.size(0) == encoded.dec_input.size(0)
        latent = encoded.dec_input
        is_cuda = latent.is_cuda
        assert type(lang_id) is int
        assert (sample is True) ^ (temperature is None)
        one_hot = None  # [] if temperature is not None else None
        n_words = self.n_words[lang_id]
        emb_layer = self.embeddings[lang_id]
        lstm_layer = self.lstm[lang_id]
        lstm_layer.flatten_parameters() # To force the weights to be contiguous in memory
        lstm_proj_layer = self.lstm_proj_layers[lang_id]
        proj_layer = self.proj[lang_id]

        # initialize generated sentences batch
        bs = latent.size(0)
        cur_len = 1
        if self.init_encoded:
            h_c = get_init_state(self.n_dec_layers, bs, self.hidden_dim, latent)
        else:
            h_c = None
        decoded = torch.LongTensor(self.max_len, bs).fill_(self.pad_index)
        decoded = decoded.cuda() if is_cuda else decoded
        decoded[0] = self.bos_index[lang_id]

        # decoding
        while cur_len < self.max_len:
            # previous word embeddings
            embeddings = emb_layer(decoded[cur_len - 1])
            embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
            if not self.init_encoded:
                embeddings = torch.cat([embeddings, latent], 1)
            lstm_output, h_c = lstm_layer(embeddings.unsqueeze(0), h_c)
            output = F.dropout(lstm_output, p=self.dropout, training=self.training).view(bs, self.hidden_dim)
            if lstm_proj_layer is not None:
                output = F.relu(lstm_proj_layer(output))
            scores = proj_layer(output).data
            assert scores.size() == (bs, n_words)

            # do no sample words not in the language vocabulary
            if self.vocab_mask_neg is not None:
                scores.index_fill_(1, self.vocab_mask_neg[lang_id], -1e30)

            # select next words: sample (Gumbel Softmax) or one-hot
            if sample:
                # if temperature is not None:
                #     gumbel = gumbel_softmax(scores, temperature, hard=True)
                #     next_words = gumbel.max(1)[1]
                #     one_hot.append(gumbel)
                # else:
                next_words = torch.multinomial((scores / temperature).exp(), 1).squeeze(1)
            else:
                next_words = scores.max(1)[1]
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words
            cur_len += 1

            # stop when there is a </s> in each sentence
            if decoded.eq(self.eos_index).sum(0).ne(0).sum() == bs:
                break

        # compute the length of each generated sentence, and
        # put some padding after the end of each sentence
        lengths = torch.LongTensor(bs).fill_(cur_len)
        for i in range(bs):
            for j in range(cur_len):
                if decoded[j, i] == self.eos_index:
                    if j + 1 < self.max_len:
                        decoded[j + 1:, i] = self.pad_index
                    lengths[i] = j + 1
                    break
            if lengths[i] == self.max_len:
                decoded[-1, i] = self.eos_index

        if one_hot is not None:
            one_hot = torch.cat([x.unsqueeze(0) for x in one_hot], 0)
            assert one_hot.size() == (cur_len - 1, bs, n_words)
        return torch.transpose(decoded[:cur_len], 0, 1), lengths, one_hot


########################################################################################
#                                                                                      #
#                                   Transformer                                        #
#                                                                                      #
########################################################################################



class TransformerEnc(nn.Module):
    """Transformer encoder."""

    ENC_ATTR = ['n_langs', 'n_words', 'dropout', 'padding_idx']

    def __init__(self, configs):
        super().__init__()
        # Configs for encoder
        configs_enc = configs['models']['encoder']['transformer']
        # language parameters
        self.n_langs = 2
        self.n_words = [configs['dataset']['vocab_size'] for _ in range(2)] # /!\ Only work if vocabulary is shared /!\
        self.pad_index = configs['dataset']['pad_index']
        # model parameters
        self.share_lang_emb = configs_enc['share_lang_emb']
        self.emb_dim = configs_enc['emb_dim']
        self.dropout = configs_enc['dropout']
        self.freeze_enc_emb = configs_enc['freeze_enc_emb']
        self.share_enc = configs['training']['share_enc'] # /!\ to change /!\
        self.n_enc_layers = configs_enc['n_enc_layers']
        self.left_pad_source = False
        # config encoder layer
        configs_enc_layer = {"encoder_embed_dim": self.emb_dim,
                             "encoder_attention_heads": configs_enc['encoder_attention_heads'],
                             "attention_dropout": configs_enc['attention_dropout'],
                             "dropout": self.dropout,
                             "relu_dropout": configs_enc['relu_dropout'],
                             "encoder_normalize_before": configs_enc['encoder_normalize_before'],
                             "encoder_ffn_embed_dim": configs_enc['transformer_ffn_embed_dim']}

        if self.share_lang_emb:
            assert len(set(self.n_words)) == 1
            print("Sharing encoder input embeddings")
            layer_0 = Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
            # Loading pre-computed embeddings weights
            self.embed_layer_name = "data\\embeddings\\simple-{}-{}_voc_{}.pt".format(
                                                    configs['dataset']['L1']['name'], 
                                                    configs['dataset']['L2']['name'],
                                                    configs['dataset']['vocab_size'])
            if os.path.exists(self.embed_layer_name):
                print('Loading pre-computed embeddings weights...')
                layer_0.load_state_dict(torch.load(
                    self.embed_layer_name
                ))
            embeddings = [layer_0 for _ in range(self.n_langs)]
        else:
            embeddings = [Embedding(n_words, self.emb_dim, padding_idx=self.pad_index) for n_words in self.n_words]
        self.embeddings = nn.ModuleList(embeddings)

        self.embed_scale = math.sqrt(self.emb_dim)
        self.embed_positions = PositionalEmbedding(
            1024, self.emb_dim, self.pad_index,
            left_pad=self.left_pad_source,
        )

        self.layers = nn.ModuleList()
        for k in range(self.n_enc_layers):
            # share top share_enc layers
            layer_is_shared = (k >= (self.n_enc_layers - self.share_enc))
            if layer_is_shared:
                print("Sharing encoder transformer parameters for layer %i" % k)

            self.layers.append(nn.ModuleList([
                # layer for first lang
                TransformerEncoderLayer(configs_enc_layer)
            ]))
            for i in range(1, self.n_langs):
                # layer for lang i
                if layer_is_shared:
                    # share layer from lang 0
                    self.layers[k].append(self.layers[k][0])
                else:
                    self.layers[k].append(TransformerEncoderLayer(configs_enc_layer))

    def forward(self, inputs, language):
        src_tokens = torch.transpose(inputs['x'], 0, 1)
        src_lengths = inputs['lengths']
        lang_id = language - 1 #/!\ Let say lang_id = language - 1 
        assert type(lang_id) is int

        embed_tokens = self.embeddings[lang_id]

        # embed tokens and positions
        x = self.embed_scale * embed_tokens(src_tokens)
        x = x.detach() if self.freeze_enc_emb else x
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.pad_index)

        # encoder layers
        for layer in self.layers:
            x = layer[lang_id](x, encoder_padding_mask)

        return {'m':LatentState(
            input_len=src_lengths,
            dec_input={
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            },
            dis_input=x,
        )}

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    @staticmethod
    def expand_encoder_out_(encoder_out, beam_size):
        T, B, C = encoder_out['encoder_out'].size()
        assert encoder_out['encoder_padding_mask'].size() == (B, T)
        encoder_out['encoder_out'] = encoder_out['encoder_out'].repeat(1, 1, beam_size).view(T, -1, C)
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].repeat(1, beam_size).view(-1, T)


class TransformerDec(nn.Module):
    """Transformer decoder."""

    DEC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), ('share_encdec_emb', False), ('share_decpro_emb', False), ('share_dec', False), 'dropout', 'eos_index', 'pad_index', 'bos_index']

    def __init__(self, configs, encoder):
        super().__init__()
        # Configs for encoder
        configs_dec = configs['models']['decoder']['transformer']
        # language parameters
        self.max_len = configs['dataset']['max_len']
        self.n_langs = 2
        self.n_words = [configs['dataset']['vocab_size'] for _ in range(2)] # /!\ Only work if vocabulary is shared /!\
        # model parameters
        self.dropout = configs_dec['dropout']
        self.share_lang_emb = configs_dec['share_lang_emb']
        self.share_encdec_emb = configs_dec['share_encdec_emb']
        self.share_decpro_emb = configs_dec['share_decpro_emb']
        self.share_output_emb = configs_dec['share_output_emb']
        self.share_dec = configs['training']['share_dec'] # /!\ To change /!\
        self.freeze_dec_emb = self.freeze_dec_emb = configs_dec['freeze_dec_emb']
        self.encoder_class = encoder.__class__
        self.beam_size = configs_dec['beam_size']
        self.length_penalty = configs_dec['length_penalty']
        self.left_pad_target = False
        self.n_dec_layers = configs_dec['n_dec_layers']
        self.emb_dim = configs_dec['emb_dim']

        # indexes
        self.eos_index = configs['dataset']['eos_index'] # End of the sentence
        self.pad_index = configs['dataset']['pad_index']
        self.bos_index = [configs['dataset']['bos{}_index'.format(k+1)] for k in range(2)] # Beginning /!\ MUST NO BE THE SAME /!\

        # config decoder layer
        configs_dec_layer = {"decoder_embed_dim": self.emb_dim,
                             "decoder_attention_heads": configs_dec['decoder_attention_heads'],
                             "attention_dropout": configs_dec['attention_dropout'],
                             "dropout": self.dropout,
                             "relu_dropout": configs_dec['relu_dropout'],
                             "encoder_normalize_before": configs['models']['encoder']['transformer']['encoder_normalize_before'],
                             "decoder_ffn_embed_dim": configs_dec['transformer_ffn_embed_dim']}

        # words allowed for generation
        self.vocab_mask_neg = None
        #self.vocab_mask_neg = args.vocab_mask_neg if len(args.vocab) > 0 else None  # TODO: implement

        # embedding layers
        if self.share_encdec_emb:
            print("Sharing encoder and decoder input embeddings")
            embeddings = encoder.embeddings
        else:
            if self.share_lang_emb:
                print("Sharing decoder input embeddings")
                layer_0 = Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
                embeddings = [layer_0 for _ in range(self.n_langs)]
            else:
                embeddings = [Embedding(n_words, self.emb_dim, padding_idx=self.pad_index) for n_words in self.n_words]
            embeddings = nn.ModuleList(embeddings)
        self.embeddings = embeddings
        self.embed_scale = math.sqrt(self.emb_dim)
        self.embed_positions = PositionalEmbedding(
            1024, self.emb_dim, self.pad_index,
            left_pad=self.left_pad_target,
        )

        self.layers = nn.ModuleList()
        for k in range(self.n_dec_layers):
            # share bottom share_dec layers
            layer_is_shared = (k < self.share_dec)
            if layer_is_shared:
                print("Sharing decoder transformer parameters for layer %i" % k)

            self.layers.append(nn.ModuleList([
                # layer for first lang
                TransformerDecoderLayer(configs_dec_layer)
            ]))
            for i in range(1, self.n_langs):
                # layer for lang i
                if layer_is_shared:
                    # share layer from lang 0
                    self.layers[k].append(self.layers[k][0])
                else:
                    self.layers[k].append(TransformerDecoderLayer(configs_dec_layer))

        # projection layers
        proj = [nn.Linear(self.emb_dim, n_words) for n_words in self.n_words]
        if self.share_decpro_emb:
            print("Sharing input embeddings and projection matrix in the decoder")
            for i in range(self.n_langs):
                proj[i].weight = self.embeddings[i].weight
            if self.share_lang_emb:
                assert self.share_output_emb
                print("Sharing decoder projection matrices")
                for i in range(1, self.n_langs):
                    proj[i].bias = proj[0].bias
        elif self.share_output_emb:
            assert self.share_lang_emb
            print("Sharing decoder projection matrices")
            for i in range(1, self.n_langs):
                proj[i].weight = proj[0].weight
                proj[i].bias = proj[0].bias
        self.proj = nn.ModuleList(proj)

    def forward(self, inputs, language, one_hot=False, incremental_state=None):
        encoded = inputs['m']
        y = torch.transpose(inputs['y'], 0, 1)[:-1] # To avoid feeding BOS in loss
        lang_id = language - 1
        assert not one_hot, 'one_hot=True has not been implemented for transformer'
        assert type(lang_id) is int

        prev_output_tokens = y  # T x B
        encoder_out = encoded.dec_input
        embed_tokens = self.embeddings[lang_id]
        proj_layer = self.proj[lang_id]

        # embed positions
        positions = self.embed_positions(prev_output_tokens, incremental_state)

        # embed tokens and positions
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[-1:, :]  # only keep last time step
        x = self.embed_scale * embed_tokens(prev_output_tokens)
        x = x.detach() if self.freeze_dec_emb else x
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn = layer[lang_id](
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state=incremental_state,
            )

        # project back to size of vocabulary
        x = proj_layer(x)

        return torch.transpose(
                torch.transpose(
                    x, 
                    0, 1),
                1, 2)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def reorder_incremental_state_(self, incremental_state, new_order):
        """Reorder incremental state.
        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

    def reorder_encoder_out_(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)

    def generate(self, inputs, language, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """

        encoded = inputs['m']
        lang_id = language - 1

        assert self.beam_size == 0
        if self.beam_size > 0:
            return self.generate_beam(encoded, lang_id, self.beam_size, self.max_len, sample, temperature)

        encoder_out = encoded.dec_input
        latent = encoder_out['encoder_out']

        x_len = encoded.input_len
        is_cuda = latent.is_cuda
        one_hot = None

        # check inputs
        assert type(lang_id) is int
        assert latent.size() == (self.max_len, x_len.size(0), self.emb_dim)
        assert (sample is True) ^ (temperature is None)

        # initialize generated sentences batch
        slen, bs = latent.size(0), latent.size(1)
        assert x_len.size(0) == bs
        cur_len = 1
        decoded = torch.LongTensor(self.max_len, bs).fill_(self.pad_index)
        unfinished_sents = torch.LongTensor(bs).fill_(1)
        lengths = torch.LongTensor(bs).fill_(1)
        if is_cuda:
            decoded = decoded.cuda()
            unfinished_sents = unfinished_sents.cuda()
            lengths = lengths.cuda()
        decoded[0] = self.bos_index[lang_id]

        incremental_state = {}
        while cur_len < self.max_len:

            # previous word embeddings
            scores = self.forward({'m': encoded, 
                                   'y': torch.transpose(decoded[:cur_len + 1], 0, 1)}, # To match the format of forward
                                   language, 
                                   one_hot, 
                                   incremental_state)
            scores = torch.transpose(torch.transpose(scores, 1, 2), 0, 1)
            scores = scores.data[-1, :, :]  # T x B x V -> B x V

            # select next words: sample or one-hot
            if sample:
                next_words = torch.multinomial((scores / temperature).exp(), 1).squeeze(1)
            else:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            lengths.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len += 1

            # stop when there is a </s> in each sentence
            if unfinished_sents.max() == 0:
                break

        if cur_len == self.max_len:
            decoded[self.max_len - 1].masked_fill_(unfinished_sents.to(torch.bool), 
                                                   self.eos_index)
        assert (decoded == self.eos_index).sum() == bs

        return torch.transpose(decoded[:cur_len], 0, 1), lengths, one_hot

    def generate_beam(self, encoded, lang_id, beam_size=20, max_len=175, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        self.encoder_class.expand_encoder_out_(encoded.dec_input, beam_size)

        x_len = encoded.input_len
        is_cuda = encoded.dec_input['encoder_out'].is_cuda
        one_hot = None

        # check inputs
        assert type(lang_id) is int
        # assert latent.size() == (x_len.max(), x_len.size(0) * beam_size, self.emb_dim)
        assert (sample is True) ^ (temperature is None)
        assert temperature is None, 'not supported'

        generator = SequenceGenerator(
            self, self.bos_index[lang_id], self.pad_index, self.eos_index,
            self.n_words[lang_id], beam_size=beam_size, maxlen=max_len, sampling=sample,
            len_penalty=self.length_penalty,
        )
        if is_cuda:
            x_len = x_len.cuda()
        results = generator.generate(x_len, encoded, lang_id)

        lengths = torch.LongTensor([sent[0]['tokens'].numel() for sent in results])
        lengths.add_(1)  # for BOS
        max_len = lengths.max()
        bsz = len(results)
        decoded = results[0][0]['tokens'].new(max_len, bsz).fill_(0)
        decoded[0, :] = self.bos_index[lang_id]
        for i, sent in enumerate(results):
            ntoks = sent[0]['tokens'].numel()  # pick the top beam result
            decoded[1:ntoks + 1, i] = sent[0]['tokens']

        return decoded, lengths, one_hot


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """
    def __init__(self, configs):
        super().__init__()
        self.embed_dim = configs['encoder_embed_dim']
        self.self_attn = MultiheadAttention(
            self.embed_dim, configs['encoder_attention_heads'],
            dropout=configs['attention_dropout'],
        )
        self.dropout = configs['dropout']
        self.relu_dropout = configs['relu_dropout']
        self.normalize_before = configs['encoder_normalize_before']
        self.fc1 = Linear(self.embed_dim, configs['encoder_ffn_embed_dim'])
        self.fc2 = Linear(configs['encoder_ffn_embed_dim'], self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""
    def __init__(self, configs):
        super().__init__()
        self.embed_dim = configs['decoder_embed_dim']
        self.self_attn = MultiheadAttention(
            self.embed_dim, configs['decoder_attention_heads'],
            dropout=configs['attention_dropout'],
        )
        self.dropout = configs['dropout']
        self.relu_dropout = configs['relu_dropout']
        self.normalize_before =configs['encoder_normalize_before']
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, configs['decoder_attention_heads'],
            dropout=configs['attention_dropout'],
        )
        self.fc1 = Linear(self.embed_dim, configs['decoder_ffn_embed_dim'])
        self.fc2 = Linear(configs['decoder_ffn_embed_dim'], self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, mask_future_timesteps=True,
            incremental_state=incremental_state, need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state, static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m





































