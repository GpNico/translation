"""
    blabla
"""

import numpy as np
from tqdm import tqdm

from gensim.models.callbacks import CallbackAny2Vec

import wandb

from datasets import load_metric, arrow_dataset

import torch

def freeze_module(module):
    """
        Freeze all parameters of module by setting
        all their requires_grad attribute to False.
    """
    for param in module.parameters():                
        param.requires_grad = False

def get_weights_name(configs):
    """
        Get weights name (!) from config file.
    """
    weights_name = "checkpoints\\{}_{}".format(
                                            configs['name'],
                                            configs['training']['prop_gold']
                                            )
    if configs['training']['denoising']:
        weights_name += '_DAE'
    if configs['training']['adversial']:
        weights_name += '_adv'
    if configs['training']['share_enc']:
        weights_name += '_share_enc'
    if configs['training']['share_dec']:
        weights_name += '_share_dec'
    weights_name += '.pt'
    
    return weights_name 

def get_configs_name(configs):
    """
        Create a name for a config file.
    """
    name = 'gold_{}'.format(configs['training']['prop_gold'])
    if configs['training']['denoising']:
        name += '_DAE'
    if configs['training']['adversial']:
        name += '_adv'
    if configs['training']['share_enc']:
        name += '_share_enc'
    if configs['training']['share_dec']:
        name += '_share_dec'

    name += '_enc_{}_bn_{}_dec_{}'.format(
                        configs['models']['encoder']['model'],
                        configs['models']['encoder']['batch_norm'],
                        configs['models']['decoder']['model']
                        )

    if configs['training']['adversial']:
        name += '_disc_{}'.format(
                    configs['models']['discriminator']['model']
                    )
    if configs['name'] in ['ContinuousSquares']:
        name += '_probs_L1_{}_L2_{}'.format(
                        configs['dataset']['L1']['probs']['type'],
                        configs['dataset']['L2']['probs']['type']
                        )
    
    return name 

def binary_accuracy(preds: torch.tensor, targets: torch.Tensor) -> torch.Tensor:
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == targets).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

######################################################################################################
# Function from Lample et al. https://github.com/facebookresearch/UnsupervisedMT

def get_mask(lengths: torch.Tensor, 
             slen: int, 
             all_words: bool, 
             expand: int =None, 
             ignore_first: bool =False, 
             batch_first: bool =False, 
             cuda: bool =True):
    """
    Create a mask of shape (slen, bs) or (bs, slen).
    """
    bs= lengths.size(0)
    mask = torch.ByteTensor(slen, bs).zero_()
    for i in range(bs):
        if all_words:
            mask[:lengths[i], i] = 1
        else:
            mask[lengths[i] - 1, i] = 1 # Give us the latent vector of the EOS token
    if expand is not None:
        assert type(expand) is int
        mask = mask.unsqueeze(2).expand(slen, bs, expand)
    if ignore_first:
        mask[0].fill_(0)
    if batch_first:
        mask = mask.transpose(0, 1)
    if cuda:
        mask = mask.cuda()
    return mask.to(torch.bool)

def get_init_state(n_dec_layers, batch_size, hidden_dim, init_state=None):
    """
    Build an initial LSTM state, with optional non-zero first layer.
    """
    init = torch.cuda.FloatTensor(n_dec_layers, batch_size, hidden_dim).zero_()
    h_0 = init.clone()
    c_0 = init.clone()
    if init_state is not None:
        assert init_state.size() == (batch_size, hidden_dim)
        h_0[0] = init_state
    return (h_0, c_0)

class WMT14Iterator():
    def __init__(self, data: arrow_dataset.Dataset, 
                       tokenizer, max_len: int = -1):
        self.num = 0
        self.data = data
        self.tokenizer = tokenizer
        self.lang = ['en', 'fr']
        self.high = 2*len(self.data)
        self.max_len = max_len

    def __iter__(self):
        self.num = 0
        self.pbar = tqdm(total=self.high)
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        self.pbar.update(1)
        if num < self.high:
            elem = self.data[num//2]['translation'][self.lang[num%2]]
            tokenized_result = self.tokenizer.encode(elem).tokens[:self.max_len]
            return tokenized_result
        raise StopIteration

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, n_epochs):
        self.epoch = 0
        self.n_epochs = n_epochs

    def on_epoch_end(self, model):
        print('FastText: epoch {}/{}'.format(self.epoch+1,
                                             self.n_epochs))
        self.epoch += 1

######################################################################################################

def log_sentences_table(sentences, predicted, golds, tokenizer, name = "prediction table"):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["source sentence", "pred translation", "gold translation"])
    for sentence, pred, gold in zip(sentences.to("cpu"), predicted.to("cpu"), golds.to("cpu")):
        sentence_text = tokenizer.decode(sentence.tolist()).replace('@@', '')
        pred_text = tokenizer.decode(pred.tolist()).replace('@@', '')
        gold_text = tokenizer.decode(gold.tolist()).replace('@@', '')
        table.add_data(sentence_text, pred_text, gold_text)
    wandb.log({name:table}, commit=False)

########################################################################################################

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_gold_params_continuous_squares(configs):

    delta = configs['dataset']['L2']['positions']['x_min'] -\
            configs['dataset']['L1']['positions']['x_min']

    x_mean_L1 = 0.5*(configs['dataset']['L1']['positions']['x_max'] +\
                     configs['dataset']['L1']['positions']['x_min'])
    y_mean_L1 = 0.5*(configs['dataset']['L1']['positions']['y_max'] +\
                     configs['dataset']['L1']['positions']['y_min'])

    a11 = configs['dataset']['M']['a11']
    a12 = configs['dataset']['M']['a12']
    a21 = configs['dataset']['M']['a21']
    a22 = configs['dataset']['M']['a22']

    a = -a11*x_mean_L1 \
        -a12*y_mean_L1
    b = -a21*x_mean_L1 \
        -a22*y_mean_L1

    enc_L1_weights = torch.tensor(
                                [[a11, a12], 
                                 [a21, a22]]
                                )

    enc_L1_bias = torch.tensor(
                            [a, b]
                            )

    det_A = a11*a22 - a21*a12
    
    dec_L1_weights = (1/det_A)*torch.tensor(
                                    [[a22, -a12], 
                                    [-a21, a11]]
                                    )

    dec_L1_bias = (-1/det_A)*torch.tensor(
                                    [a22*a - a12*b, 
                                     -a21*a + a11*b]
                                    )
    

    dec_L2_weights = (1/det_A)*torch.tensor(
                                    [[a22, -a12], 
                                    [-a21, a11]]
                                    )

    dec_L2_bias = (-1/det_A)*torch.tensor(
                                    [a22*a - a12*b - det_A*delta, 
                                     -a21*a + a11*b]
                                    )

    enc_L2_weights = torch.tensor(
                                [[a11, a12], 
                                 [a21, a22]]
                                )

    
    b_prime = (dec_L2_bias[1] - (dec_L2_weights[1,0]/dec_L2_weights[0,0])*dec_L2_bias[0])*\
              (1/(dec_L2_weights[0,1]*dec_L2_weights[1,0]/dec_L2_weights[0,0] - dec_L2_weights[1,1]))
    a_prime = (-dec_L2_weights[0,1]*b_prime - dec_L2_bias[0])/dec_L2_weights[0,0]

    enc_L2_bias = torch.tensor(
                            [a_prime,
                             b_prime]
                            )

    return (enc_L1_weights, 
            enc_L1_bias, 
            enc_L2_weights, 
            enc_L2_bias,
            dec_L1_weights, 
            dec_L1_bias,
            dec_L2_weights, 
            dec_L2_bias) 

def get_gold_params_gaussian_mixture(configs):

    delta = configs['dataset']['L']['delta']

    mus1 = np.array(configs['dataset']['L']['mus1'])



    x_mean_L1 = mus1[:,0].mean()
    y_mean_L1 = mus1[:,1].mean()

    a11 = configs['dataset']['M']['a11']
    a12 = configs['dataset']['M']['a12']
    a21 = configs['dataset']['M']['a21']
    a22 = configs['dataset']['M']['a22']

    a = -a11*x_mean_L1 \
        -a12*y_mean_L1
    b = -a21*x_mean_L1 \
        -a22*y_mean_L1

    enc_L1_weights = torch.tensor(
                                [[a11, a12], 
                                 [a21, a22]],
                                 dtype = torch.float32
                                )

    enc_L1_bias = torch.tensor(
                            [a, b],
                            dtype = torch.float32
                            )

    det_A = a11*a22 - a21*a12
    
    dec_L1_weights = (1/det_A)*torch.tensor(
                                    [[a22, -a12], 
                                    [-a21, a11]],
                                    dtype = torch.float32
                                    )

    dec_L1_bias = (-1/det_A)*torch.tensor(
                                    [a22*a - a12*b, 
                                     -a21*a + a11*b],
                                     dtype = torch.float32
                                    )
    

    dec_L2_weights = (1/det_A)*torch.tensor(
                                    [[a22, -a12], 
                                    [-a21, a11]],
                                    dtype = torch.float32
                                    )

    dec_L2_bias = (-1/det_A)*torch.tensor(
                                    [a22*a - a12*b - det_A*delta, 
                                     -a21*a + a11*b],
                                    dtype = torch.float32
                                    )

    enc_L2_weights = torch.tensor(
                                [[a11, a12], 
                                 [a21, a22]],
                                dtype = torch.float32
                                )

    
    b_prime = (dec_L2_bias[1] - (dec_L2_weights[1,0]/dec_L2_weights[0,0])*dec_L2_bias[0])*\
              (1/(dec_L2_weights[0,1]*dec_L2_weights[1,0]/dec_L2_weights[0,0] - dec_L2_weights[1,1]))
    a_prime = (-dec_L2_weights[0,1]*b_prime - dec_L2_bias[0])/dec_L2_weights[0,0]

    enc_L2_bias = torch.tensor(
                            [a_prime,
                             b_prime],
                             dtype = torch.float32
                            )

    return (enc_L1_weights, 
            enc_L1_bias, 
            enc_L2_weights, 
            enc_L2_bias,
            dec_L1_weights, 
            dec_L1_bias,
            dec_L2_weights, 
            dec_L2_bias)
