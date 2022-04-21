"""
    blabla
"""    

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.languages import Languages


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
                       languages: Languages, # Needed to convert ids to sentences /!\ Not clean
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
        # Used to generate sentences
        self.languages = languages
        
        # Either ways we train those
        self.s1 = Decoder(model = configs['dec_model'],
                          dim_in = configs['dim_M'],
                          dim_out = configs['dec_dim_out'])
        self.r1 = Encoder(model = configs['enc_model'],
                          dim_in = configs['dim_L'],
                          dim_out = configs['dim_M'])
        self.s2 = Decoder(model = configs['dec_model'],
                          dim_in = configs['dim_M'],
                          dim_out = configs['dec_dim_out'])
        self.r2 = Encoder(model = configs['enc_model'],
                          dim_in = configs['dim_L'],
                          dim_out = configs['dim_M'])


        # Gold 
        #self.R1 = Receiver1(configs)
        #self.S2 = Sender2(configs)
        #self.R2 = Receiver2(configs)
        #self.S1 = Sender1(configs)
        
    def denoising_autoencoding(self, sentences: torch.Tensor, 
                                     language: int) -> torch.Tensor:
        """
            Forward pass of the DAE objective.
        """

        if language == 1:
            m = self.r1(sentences)
            recons = self.s1(m)
            return recons
        elif language == 2:
            m = self.r2(sentences)
            recons = self.s2(m)
            return recons
        else:
            raise Exception("language should be an int equals to 1 or 2.")

    def back_translation(self, sentences: torch.Tensor,
                               language: int) -> torch.Tensor:
        """
            Forward pass of the BT objective.
        """
        # Language L1
        if language == 1:
            if self.train_all:
                # Not working now in stochastic setup because 
                # of the generation (cannot backpropagate)
                m = self.r1(sentences)
                if self.stochastic:
                    # Need Gumbel-Softmax
                    sentences_prime_probs = F.softmax(self.s2(m), dim = -1)
                else:
                    sentences_prime = self.s2(m)
            else:
                with torch.no_grad():
                    m = self.r1(sentences)
                    if self.stochastic:
                        # Compute probability distribution
                        sentences_prime_probs = F.softmax(self.s2(m), dim = -1)
                        # Generate sentence from it
                        sentences_prime = self._generate_from_probs(sentences_prime_probs).to(self.device)
                    else:
                        sentences_prime = self.s2(m)
            m_prime = self.r2(sentences_prime)
            sentences_recons = self.s1(m_prime)
            return sentences_recons
        # Language L2
        elif language == 2:
            if self.train_all:
                # Not working now in stochastic setup because 
                # of the generation (cannot backpropagate)
                m = self.r2(sentences)
                if self.stochastic:
                    # Need Gumbel-Softmax
                    sentences_prime_probs = self.s1(m)
                else:
                    sentences_prime = self.s1(m)
            else:
                with torch.no_grad():
                    m = self.r2(sentences)
                    if self.stochastic:
                        # Compute probability distribution
                        sentences_prime_probs = F.softmax(self.s1(m), dim = -1)
                        # Generate sentence from it
                        sentences_prime = self._generate_from_probs(sentences_prime_probs).to(self.device)
                    else:
                        sentences_prime = self.s1(m)
            m_prime = self.r1(sentences_prime)
            sentences_recons = self.s2(m_prime)
            return sentences_recons
        else:
            raise Exception("language should be an int equals to 1 or 2.")



        pass

    def translate(self, sentences: torch.Tensor,
                        language: int) -> torch.Tensor:
        """
            Translate sentences (from language)
            according to :
                - T_L^{12} = s2 o T_M_{12} o r1
                - T_L^{21} = s1 o T_M_{21} o r2
        """

        # In the stochastic scenario translation is a 
        # probability distribution.
        # Do we take an argmax here ?

        if language == 1:
            m = self.r1(sentences)
            translation = self.s2(m)
            return translation
        elif language == 2:
            m = self.r2(sentences)
            translation = self.s1(m)
            return translation
        else:
            raise Exception("language should be an int equals to 1 or 2.")

    def _generate_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
            Args:
                probs (tensor) shape [batch_size, sentnces_num] (in toy examples)
                                     [batch_size, n_tokens, voc_size] (in real examples)
        """
        # Not optimal at all ! Especially because of convert_id_to_coords
        # which uses a for loop !!
        # In the toy examples setup
        sentences_ids = torch.multinomial(probs, 1).squeeze(1) # [batch_size]
        # Sampled sentences
        generated_sentences = self.languages.convert_id_to_coords(sentences_ids)
        return generated_sentences


class Encoder(nn.Module):
    """
        enc_i : L_i -> M_i
        language to meaning
    """
    def __init__(self, model: str = 'linear',
                       dim_in: int = 3,
                       dim_out: int = 2
                       ) -> None:
        super(Encoder, self).__init__()
        
        # To Modify !

        if model == 'linear':
            self.enc = nn.Linear(dim_in, dim_out, bias = True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        m = self.enc(input)
        return m

class Decoder(nn.Module):
    """
        dec_i : M_i -> L_i
        meaning to language
    """
    def __init__(self, model,
                       dim_in: int = 2,
                       dim_out: int = 3
                       ) -> None:
        super(Decoder, self).__init__()
        
        if model == 'linear':
            self.dec = nn.Linear(dim_in, dim_out, bias = True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = self.dec(input)
        return l
    
        


class Receiver1(nn.Module):
    """
        R1 : L1 -> M1
        language to meaning (gold label)
    """
    def __init__(self, configs):
        super(Receiver1, self).__init__()
        
        if configs['name'] == 'v0':
            self.enc = nn.Linear(configs['dim_in'], configs['dim_out'], bias = False)
            self.enc.weight = torch.nn.Parameter(torch.tensor([[2., 1.], [0.5, -1.]]))
            freeze_module(self.enc) # We do not train the gold label
            
        elif configs['name'] == 'v1':
            self.enc = ComputeMeaningV1(configs, language = 1)
        
    def forward(self, l):
        m = self.enc(l)
        return m


class Sender1(nn.Module):
    """
        S1 : M1 -> L1
        meaning to language (gold label)
    """
    def __init__(self, configs):
        super(Sender1, self).__init__()
        
        if configs['name'] == 'v0':
            self.dec = nn.Linear(configs['dim_out'], configs['dim_in'], bias = False)
            self.dec.weight = torch.nn.Parameter((1/(-2.5))*torch.tensor([[-1., -1.], [-0.5, 2.]]))
            freeze_module(self.dec) # We do not train the gold label
            
        elif configs['name'] == 'v1':
            self.dec = SampleSynonymsV1(configs, language = 1)
        
    def forward(self, m):
        l = self.dec(m)
        return l