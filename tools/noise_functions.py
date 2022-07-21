""""
    Blabla
"""
import numpy as np

import torch

class Noiser:
    """
        Class that wrapps all the noise
        functions.
    """

    def __init__(self, noise_func: str = "identity",
                       noise_intensity = None,
                       tokenizer = None) -> None:

        self.noise_intensity = noise_intensity

        if noise_func == 'identity':
            self.noise = self.identity
        elif noise_func == 'gaussian':
            self.noise = self.gaussian
        elif noise_func == 'shuffle_drop_blank':
            self.tokenizer = tokenizer
            assert self.tokenizer is not None
            self.init_bpe()
            self.noise = self.shuffle_drop_blank
        else:
            print("No correct noise functuion indicated. Selecting identity.")
            self.noise = self.identity

    def identity(self, inputs: dict) -> torch.Tensor:
        """
            No noise.
        """
        x, lengths = inputs['x'], inputs['lengths']
        return x, lengths

    def gaussian(self, inputs: dict) -> torch.Tensor:
        """
            Add gaussian perturbation.
        """
        x, lengths = inputs['x'], inputs['lengths']
        return x + self.noise_intensity * torch.normal(mean = 0., 
                                                       std = torch.ones_like(x)), lengths

    def shuffle_drop_blank(self, inputs: dict) -> torch.Tensor:
        x, lengths, language = torch.transpose(inputs['x'], 0, 1), inputs['lengths'], inputs['language']
        """
        print("vanilla : ", x.shape)
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[0]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[1]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[2]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[3]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[4]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[5]))
        """
        x, lengths = self.word_shuffle(x, lengths)
        """
        print("shuffled : ", x.shape)
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[0]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[1]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[2]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[3]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[4]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[5]))
        """
        x, lengths = self.word_dropout(x, lengths, language)
        """
        print("dropout : ", x.shape)
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[0]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[1]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[2]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[3]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[4]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[5]))
        """
    
        x, lengths = self.word_blank(x, lengths, language)
        """
        print("dropout : ", x.shape)
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[0]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[1]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[2]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[3]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[4]))
        print("sentence : ", self.tokenizer.decode(torch.transpose(x, 0, 1).tolist()[5]))
        print("")
        print("sentence : ", [self.tokenizer.id_to_token(token) for token in torch.transpose(x, 0, 1).tolist()[0]])
        print("sentence : ", [self.tokenizer.id_to_token(token) for token in torch.transpose(x, 0, 1).tolist()[1]])
        print("sentence : ", [self.tokenizer.id_to_token(token) for token in torch.transpose(x, 0, 1).tolist()[2]])
        exit(0)
        """
        
        return torch.transpose(x, 0, 1) , lengths


    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.noise_intensity['word_shuffle'] == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, 
                                  self.noise_intensity['word_shuffle'], 
                                  size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[x]
        word_idx = bpe_end.cumsum(0) - 1

        assert self.noise_intensity['word_shuffle'] >= 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l, language):
        """
        Randomly drop input words.
        """
        x_length = x.size(0)
        if self.noise_intensity['word_dropout'] == 0:
            return x, l
        assert 0 < self.noise_intensity['word_dropout']  < 1

        # define words to drop
        bos_index = self.tokenizer.token_to_id('[BOS{}]'.format(language))
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.noise_intensity['word_dropout'] 
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[x]
        word_idx = bpe_end.cumsum(0) - 1

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.tokenizer.token_to_id('[EOS]')
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.tokenizer.token_to_id('[EOS]'))
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.tokenizer.token_to_id('[EOS]')
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(x_length, l2.size(0)).fill_(self.tokenizer.token_to_id('[PAD]'))
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l, language):
        """
        Randomly blank input words.
        """
        x_length = x.size(0)
        if self.noise_intensity['word_blank'] == 0:
            return x, l
        assert 0 < self.noise_intensity['word_blank'] < 1

        # define words to blank
        bos_index = self.tokenizer.token_to_id('[BOS{}]'.format(language))
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.noise_intensity['word_blank']
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[x]
        word_idx = bpe_end.cumsum(0) - 1

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.tokenizer.token_to_id('[EOS]')
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.tokenizer.token_to_id('[BLANK]') for j, w in enumerate(words)] # Don't have blank index so using x instead... Maybe need to retokenize everything
            new_s.append(self.tokenizer.token_to_id('[EOS]'))
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.tokenizer.token_to_id('[EOS]')
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(x_length, l.size(0)).fill_(self.tokenizer.token_to_id('[PAD]'))
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def init_bpe(self):
        """
        Index BPE words.
        """
        self.bpe_end = np.array(
                    [not self.tokenizer.id_to_token(i).startswith('@@') \
                        for i in range(self.tokenizer.get_vocab_size())]
                    )