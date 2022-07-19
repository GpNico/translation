"""
    blabla
"""   
import numpy as np
import os
import unicodedata
import re
from gensim.models.fasttext import FastText

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

from sklearn.model_selection import train_test_split

from tools.helpers import WMT14Iterator, callback


class SentencesDataset(Dataset):
    """
        Create Pytorch Dataset from given inputs.

    """
    def __init__(self, data,
                       tokenizer = None,
                       name = None, 
                       language = None,
                       max_len: int = -1,
                       max_data_size: int = -1):
        self.data = data
        self.tokenizer = tokenizer
        self.name = name
        self.language = language
        # Max length of every tokenized sentence
        self.max_len = max_len
        # Max size of the dataset
        self.max_data_size = max_data_size

    def __len__(self) -> int:
        if self.name is not None:
            if self.name in ['WMT14', 'SimpleEN_FR', 'PCFG']:
                if self.max_data_size == -1:
                    return len(self.data)
                return min(self.max_data_size , len(self.data))
        return self.data.shape[0]

    def __getitem__(self, item: int):
        if self.tokenizer is not None:
            if self.name is not None:
                if self.name in ['WMT14']:
                    if self.language == 1:
                        gold_trad, sentence = self.data[item]['translation'].values()
                    elif self.language == 2:
                        sentence, gold_trad = self.data[item]['translation'].values()
                elif self.name in ['SimpleEN_FR', 'PCFG']:
                    if self.language == 1:
                        sentence, gold_trad = self.data[item]
                    elif self.language == 2:
                        gold_trad, sentence = self.data[item]
                else:
                    sentence = self.data[item]
                    gold_trad = None

            bos1 = '[BOS1]' if self.language == 1 else '[BOS2]'
            bos2 = '[BOS2]' if self.language == 1 else '[BOS1]'
            eos = '[EOS]'

            input_ids = self.tokenizer.encode(bos1 + sentence + eos).ids
            length = min(len(input_ids), self.max_len)
            input_ids = input_ids + [self.tokenizer.token_to_id('[PAD]') \
                                     for _ in range(self.max_len - length)]
            input_ids = torch.tensor(input_ids[:self.max_len],
                                     dtype = torch.int)
            if length == self.max_len:
                input_ids[-1] = self.tokenizer.token_to_id('[EOS]')

            gold_input_ids = self.tokenizer.encode(bos2 + gold_trad + eos).ids
            gold_length = min(len(gold_input_ids), self.max_len)
            gold_input_ids = gold_input_ids + [self.tokenizer.token_to_id('[PAD]') \
                                               for _ in range(self.max_len - len(gold_input_ids))]
            gold_input_ids = torch.tensor(gold_input_ids[:self.max_len],
                                          dtype = torch.int)

            if gold_length == self.max_len:
                gold_input_ids[-1] = self.tokenizer.token_to_id('[EOS]')

            item_dict = {'sentences': input_ids,
                         'gold translation': gold_input_ids,
                         'lengths': length,
                         'gold lengths': gold_length}
        else:
            item_dict = {'sentences': self.data[item],
                         'lengths': -1}
        return item_dict


class Languages:
    """
        Parent Class for Languages.
    """

    def __init__(self, configs) -> None:
        self.configs = configs
        self.name = None
        # Sentences (= datasets)
        self.L1_data = None
        self.L2_data = None

        # Might be of use
        self.tokenizer = None

        # Max data size
        self.max_size = -1

    def _get_datasets(self, language: int):
        """
            Construct datasets for train, Validation
            and test.
        """
        assert language == 1 or language == 2

        data_train, data_valid, data_test = self._train_valid_test_split(language)

        train_dataset = SentencesDataset(data = data_train,
                                         tokenizer = self.tokenizer,
                                         name = self.name,
                                         language = language,
                                         max_len = self.configs['dataset']['max_len'],
                                         max_data_size = self.max_size)
        valid_dataset = SentencesDataset(data = data_valid,
                                         tokenizer = self.tokenizer,
                                         name = self.name,
                                         language = language,
                                         max_len = self.configs['dataset']['max_len'],
                                         max_data_size = self.max_size)
        test_dataset = SentencesDataset(data = data_test,
                                        tokenizer = self.tokenizer,
                                        name = self.name,
                                        language = language,
                                        max_len = self.configs['dataset']['max_len'],
                                        max_data_size = self.max_size)

        #print(test_dataset.__getitem__(0))

        return train_dataset, valid_dataset, test_dataset

    def get_dataloaders(self, language: int,
                              batch_size: int,
                              prop_gold: int):
        """
            Construct dataloaders for train, validation
            and test. 
        """
        assert language == 1 or language == 2

        train_dataset, valid_dataset, test_dataset = self._get_datasets(language)

        if prop_gold > 0 and prop_gold < 1:
            train_dataset, train_dataset_gold = self._train_gold_split(train_dataset,
                                                                       prop_gold)

            train_gold_loader = DataLoader(train_dataset_gold, 
                                           batch_size = batch_size, 
                                           shuffle = True, 
                                           num_workers = 0, 
                                           drop_last = True)

        train_loader = DataLoader(train_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  num_workers = 0, 
                                  drop_last = True)

        valid_loader = DataLoader(valid_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = False, 
                                  num_workers = 0, 
                                  drop_last = True)

        test_loader = DataLoader(test_dataset, 
                                 batch_size = batch_size, 
                                 shuffle = True, 
                                 num_workers = 0, 
                                 drop_last = True)

        if prop_gold > 0 and prop_gold < 1:
            return train_loader, train_gold_loader, valid_loader, test_loader
        elif prop_gold == 1:
            return None, train_loader, valid_loader, test_loader
        else:
            return train_loader, None, valid_loader, test_loader


    def _train_valid_test_split(self, language: int):
        """
           Split sentences in three train, valiation
           and test sets respectively.
        """
        assert language == 1 or language == 2

        if language == 1:
            data = self.L1_data
        elif language == 2:
            data = self.L2_data

        data_train_valid, data_test = train_test_split(data, test_size=0.2, random_state=1)

        data_train, data_valid, = train_test_split(data_train_valid, test_size=0.25, random_state=1)

        return data_train, data_valid, data_test

    def _train_gold_split(self, train_dataset: SentencesDataset,
                                prop_gold: int):

        train_size_no_gold = int((1-prop_gold) * len(train_dataset))
        train_size_gold = len(train_dataset) - train_size_no_gold
        train_dataset_no_gold, train_dataset_gold = torch.utils.data.random_split(train_dataset, 
                                                                                  [train_size_no_gold, 
                                                                                   train_size_gold])

        return train_dataset_no_gold, train_dataset_gold


    def _create_data(self, language: int) -> torch.Tensor:
        """
            Create data of the language i.
            Specific to each languages.
            TBD
        """
        raise Exception("_create_sentences is not implemented!")


class WMT14(Languages):
    """
        Real Data.
    """
    def __init__(self, configs):
        super(WMT14, self).__init__(configs)

        # Name
        self.name = 'WMT14'

        # language names
        self.L1_name = configs['dataset']['L1']['name']
        self.L2_name = configs['dataset']['L2']['name']

        # Max size
        self.max_size = configs['dataset']['size']

        # Load the dataset
        self.data = load_dataset("wmt14", 
                                 "{}-{}".format(
                                        self.L1_name,
                                        self.L2_name
                                        )
                                )

        # Load tokenizer and/or train tokenizer
        if configs['dataset']['joint_tokenization']:
            tokenizer_name = "data\\tokenizer\\wmt14-{}-{}_voc_{}.json".format(
                                                                    self.L1_name, 
                                                                    self.L2_name,
                                                                    configs['dataset']['vocab_size'])

            if os.path.exists(tokenizer_name):
                print("Found tokenizer at {}".format(tokenizer_name))
                self.tokenizer = Tokenizer.from_file(tokenizer_name)
            else:
                print("Training new tokenizer...")
                self.tokenizer = Tokenizer(
                                        BPE(
                                            unk_token="[UNK]"
                                        )
                                    )
                self.tokenizer.pre_tokenizer = Whitespace()

                special_tokens = [None, None, None, None]
                special_tokens[configs['dataset']['pad_index']] = '[PAD]'
                special_tokens[configs['dataset']['bos1_index']] = '[BOS1]'
                special_tokens[configs['dataset']['bos2_index']] = '[BOS2]'
                special_tokens[configs['dataset']['eos_index']] = '[EOS]'
                trainer = BpeTrainer(
                                vocab_size = configs['dataset']['vocab_size'],
                                special_tokens = special_tokens,
                                continuing_subword_prefix = '@@',
                                end_of_word_suffix = '</w>'
                                )
                self.tokenizer.train_from_iterator(
                                        self._batch_iterator(), 
                                        trainer=trainer,
                                        length = 2*len(self.data['train'])
                                        )
                self.tokenizer.decoder = decoders.BPEDecoder(suffix = '</w>')

                self.tokenizer.save(tokenizer_name)

                print("Done! New tokenizer saved at {}".format(tokenizer_name))

            # if init embedding
            self.embed_layer_name = "data\\embeddings\\wmt14-{}-{}_voc_{}.pt".format(
                                                    self.L1_name, 
                                                    self.L2_name,
                                                    self.configs['dataset']['vocab_size'])
            if os.path.exists(self.embed_layer_name):
                print("Found Embedding Layer at {}".format(
                    self.embed_layer_name
                ))
            else:
                self._compute_fasttext_embeddings()
                exit(0)
        else:
            tokenizer_names = ["data\\tokenizer\\wmt14-{}.json".format(name)\
                                 for name in [self.L1_name, self.L2_name]]
            raise Exception("Separate Tokenizaton is Not Implemented Yet!")

    def _compute_fasttext_embeddings(self):
        print("Computing FastText embeddings...")

        # Load and tokeniz the corpus
        tokenized_iterator = WMT14Iterator(data = self.data['train'],
                                           tokenizer = self.tokenizer,
                                           max_len = self.configs['dataset']['max_len'])

        # Params
        enc_model = self.configs['models']['encoder']['model']
        emb_dim = self.configs['models']['encoder'][enc_model]['emb_dim']
        # Taken from Lample
        min_count = 0
        window_size = 5
        negative = 10
        n_epochs = 10

        # Train model
        print("Training FastText model...")
        ft_model = FastText(tokenized_iterator,
                            vector_size=emb_dim,
                            window=window_size,
                            min_count=min_count,
                            negative=negative,
                            sg=1, # SkipGram
                            epochs=n_epochs,
                            callbacks=[callback(n_epochs = n_epochs)])

        # Get embedding matrix
        embed_layer = nn.Embedding(self.configs['dataset']['vocab_size'], 
                                   emb_dim, 
                                   padding_idx = self.configs['dataset']['pad_index'])
        matrix_embedd = embed_layer.weight.detach() # Not that as this a pointer, changing matrix_embed will change embed_layer weights
        wv = ft_model.wv
        key_to_index = wv.key_to_index
        print("Computing embedding for {} tokens over {}".format(
            len(key_to_index),
            self.configs['dataset']['vocab_size']
        ))
        #index_to_key = {v:k for k,v in key_to_index.items()}
        ignored_tokens = 0 # Token that doen't have embedding
        for k in range(self.configs['dataset']['vocab_size']):
            token = self.tokenizer.id_to_token(k)
            if token in key_to_index.keys():
                # Token has an embedding
                matrix_embedd[k] = torch.tensor(wv[token])
            else:
                ignored_tokens += 1

        print("Ignored tokens : %s " % ignored_tokens)

        # Saving
        torch.save(embed_layer.state_dict(), self.embed_layer_name)
        print("Embedding layer saved at {}".format(self.embed_layer_name))

    def _train_valid_test_split(self, language: int):
        data_train = self.data['train']
        data_valid = self.data['validation']
        data_test = self.data['test']

        return (data_train, data_valid, data_test)

    def _batch_iterator(self, batch_size: int = 1024):
        for i in range(0, len(self.data['train']), batch_size):
            batch = [elem[self.L1_name] for elem in self.data['train'][i : i + batch_size]['translation']] +\
                    [elem[self.L2_name] for elem in self.data['train'][i : i + batch_size]['translation']]
            yield batch

class PCFG(Languages):

    def __init__(self, configs):
        super(PCFG, self).__init__(configs)

        # Name
        self.name = 'PCFG'

        # language names
        self.L1_name = configs['dataset']['L1']['name']
        self.L2_name = configs['dataset']['L2']['name']

        # Max size
        self.max_size = configs['dataset']['size']

        # Load the dataset
        self.data = self.load_data() # List of pairs
        print("Loaded %s pairs!" % len(self.data))

        # Load tokenizer and/or train tokenizer
        if configs['dataset']['joint_tokenization']:
            tokenizer_name = os.path.join("data",
                                          "tokenizer",
                                          "pcfg-{}-{}_voc_{}.json".format(
                                                                    self.L1_name, 
                                                                    self.L2_name,
                                                                    configs['dataset']['vocab_size'])
                                        )

            if os.path.exists(tokenizer_name):
                print("Found tokenizer at {}".format(tokenizer_name))
                self.tokenizer = Tokenizer.from_file(tokenizer_name)
            else:
                print("Training new tokenizer...")
                self.tokenizer = Tokenizer(
                                        BPE()
                                    )
                self.tokenizer.pre_tokenizer = Whitespace()

                special_tokens = [None, None, None, None]
                special_tokens[configs['dataset']['pad_index']] = '[PAD]'
                special_tokens[configs['dataset']['bos1_index']] = '[BOS1]'
                special_tokens[configs['dataset']['bos2_index']] = '[BOS2]'
                special_tokens[configs['dataset']['eos_index']] = '[EOS]'
                trainer = BpeTrainer(
                                vocab_size = configs['dataset']['vocab_size'],
                                special_tokens = special_tokens,
                                continuing_subword_prefix = '@@',
                                end_of_word_suffix = '</w>'
                                )
                self.tokenizer.train_from_iterator(
                                        self._batch_iterator(), 
                                        trainer=trainer,
                                        length = 2*len(self.data)
                                        )
                self.tokenizer.decoder = decoders.BPEDecoder(suffix = '</w>')

                self.tokenizer.save(tokenizer_name)

                print("Done! New tokenizer saved at {}".format(tokenizer_name))

            # if init embedding
            self.embed_layer_name = os.path.join("data",
                                                 "embeddings",
                                                 "pcfg-{}-{}_voc_{}.pt".format(
                                                    self.L1_name, 
                                                    self.L2_name,
                                                    self.configs['dataset']['vocab_size'])
                                                )
            if os.path.exists(self.embed_layer_name):
                print("Found Embedding Layer at {}".format(
                    self.embed_layer_name
                ))
            else:
                self._compute_fasttext_embeddings()
        else:
            tokenizer_names = ["data\\tokenizer\\pcfg-{}.json".format(name)\
                                 for name in [self.L1_name, self.L2_name]]
            raise Exception("Separate Tokenizaton is Not Implemented Yet!")

    def _batch_iterator(self, batch_size: int = 1024):
        for i in range(0, len(self.data), batch_size):
            batch = [elem[0] for elem in self.data[i : i + batch_size]] +\
                    [elem[1] for elem in self.data[i : i + batch_size]]
            yield batch

    def _train_valid_test_split(self, language: int):
        """
           Split sentences in three train, valiation
           and test sets respectively.
        """

        data_train_valid, data_test = train_test_split(self.data, test_size=0.01, random_state=1) # To remove when performing analysis 

        data_train, data_valid, = train_test_split(data_train_valid, test_size=0.25, random_state=1)

        return data_train, data_valid, data_test

    def _compute_fasttext_embeddings(self):
        print("Computing FastText embeddings...")

        # Load and tokeniz the corpus
        token_tokenized_corpus = self._load_tokenized_corpus()

        # Params
        enc_model = self.configs['models']['encoder']['model']
        emb_dim = self.configs['models']['encoder'][enc_model]['emb_dim']
        # Taken from Lample
        min_count = 0
        window_size = 5
        negative = 10
        n_epochs = 10

        # Train model
        print("Training FastText model...")
        ft_model = FastText(token_tokenized_corpus,
                            vector_size=emb_dim,
                            window=window_size,
                            min_count=min_count,
                            negative=negative,
                            sg=1, # SkipGram
                            epochs=n_epochs,
                            callbacks=[callback(n_epochs = n_epochs)])

        # Get embedding matrix
        embed_layer = nn.Embedding(self.configs['dataset']['vocab_size'], 
                                   emb_dim, 
                                   padding_idx = self.configs['dataset']['pad_index'])
        matrix_embedd = embed_layer.weight.detach() # Not that as this a pointer, changing matrix_embed will change embed_layer weights
        wv = ft_model.wv
        key_to_index = wv.key_to_index
        print("Computing embedding for {} tokens over {}".format(
            len(key_to_index),
            self.configs['dataset']['vocab_size']
        ))
        #index_to_key = {v:k for k,v in key_to_index.items()}
        ignored_tokens = 0 # Token that doen't have embedding
        for k in range(self.configs['dataset']['vocab_size']):
            token = self.tokenizer.id_to_token(k)
            if token in key_to_index.keys():
                # Token has an embedding
                matrix_embedd[k] = torch.tensor(wv[token])
            else:
                ignored_tokens += 1

        print("Ignored tokens : %s " % ignored_tokens)

        # Saving
        torch.save(embed_layer.state_dict(), self.embed_layer_name)
        print("Embedding layer saved at {}".format(self.embed_layer_name))

    def _load_tokenized_corpus(self):
        print("Tokenizing corpus...")
        token_tokenized_corpus = []
        for k in range(len(self.data)):
            token_tokenized_corpus.append(
                self.tokenizer.encode(self.data[k][0]).tokens
            )
            token_tokenized_corpus.append(
                self.tokenizer.encode(self.data[k][1]).tokens
            )
        return token_tokenized_corpus

    def load_data(self):
        with open(os.path.join('data','artificial_grammar','permuted_samples','sample_{}.txt'.format(self.L1_name))) as f:
            lines_1 = f.readlines()
        with open(os.path.join('data','artificial_grammar','permuted_samples','sample_{}.txt'.format(self.L2_name))) as f:
            lines_2 = f.readlines()
        assert len(lines_1) == len(lines_2)
        pairs = []
        for k in range(len(lines_1)):
            pairs.append([lines_1[k].lower().strip(), 
                          lines_2[k].lower().strip()])
        return pairs

class SimpleEN_FR(Languages):
    """
        Data from PyTorch Tutorial : https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 
        Simple sentences starting with :
            "i am ", "i m ",
            "he is", "he s ",
            "she is", "she s ",
            "you are", "you re ",
            "we are", "we re ",
            "they are", "they re "
    """
    ENG_PREFIXES = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )

    def __init__(self, configs):
        super(SimpleEN_FR, self).__init__(configs)

        # Name
        self.name = 'SimpleEN_FR'

        # language names
        self.L1_name = configs['dataset']['L1']['name']
        self.L2_name = configs['dataset']['L2']['name']

        # Max size
        self.max_size = configs['dataset']['size']

        # Load the dataset
        self.data = self.load_data() # List of pairs
        print("Loaded %s pairs!" % len(self.data))

        # Load tokenizer and/or train tokenizer
        if configs['dataset']['joint_tokenization']:
            tokenizer_name = "data\\tokenizer\\simple-{}-{}_voc_{}.json".format(
                                                                    self.L1_name, 
                                                                    self.L2_name,
                                                                    configs['dataset']['vocab_size'])

            if os.path.exists(tokenizer_name):
                print("Found tokenizer at {}".format(tokenizer_name))
                self.tokenizer = Tokenizer.from_file(tokenizer_name)
            else:
                print("Training new tokenizer...")
                self.tokenizer = Tokenizer(
                                        BPE()
                                    )
                self.tokenizer.pre_tokenizer = Whitespace()

                special_tokens = [None, None, None, None]
                special_tokens[configs['dataset']['pad_index']] = '[PAD]'
                special_tokens[configs['dataset']['bos1_index']] = '[BOS1]'
                special_tokens[configs['dataset']['bos2_index']] = '[BOS2]'
                special_tokens[configs['dataset']['eos_index']] = '[EOS]'
                trainer = BpeTrainer(
                                vocab_size = configs['dataset']['vocab_size'],
                                special_tokens = special_tokens,
                                continuing_subword_prefix = '@@',
                                end_of_word_suffix = '</w>'
                                )
                self.tokenizer.train_from_iterator(
                                        self._batch_iterator(), 
                                        trainer=trainer,
                                        length = 2*len(self.data)
                                        )
                self.tokenizer.decoder = decoders.BPEDecoder(suffix = '</w>')

                self.tokenizer.save(tokenizer_name)

                print("Done! New tokenizer saved at {}".format(tokenizer_name))

            # if init embedding
            self.embed_layer_name = "data\\embeddings\\simple-{}-{}_voc_{}.pt".format(
                                                    self.L1_name, 
                                                    self.L2_name,
                                                    self.configs['dataset']['vocab_size'])
            if os.path.exists(self.embed_layer_name):
                print("Found Embedding Layer at {}".format(
                    self.embed_layer_name
                ))
            else:
                self._compute_fasttext_embeddings()
        else:
            tokenizer_names = ["data\\tokenizer\\simple_en_fr-{}.json".format(name)\
                                 for name in [self.L1_name, self.L2_name]]
            raise Exception("Separate Tokenizaton is Not Implemented Yet!")

    def _batch_iterator(self, batch_size: int = 1024):
        for i in range(0, len(self.data), batch_size):
            batch = [elem[0] for elem in self.data[i : i + batch_size]] +\
                    [elem[1] for elem in self.data[i : i + batch_size]]
            yield batch

    def _train_valid_test_split(self, language: int):
        """
           Split sentences in three train, valiation
           and test sets respectively.
        """

        data_train_valid, data_test = train_test_split(self.data, test_size=0.01, random_state=1) # To remove when performing analysis 

        data_train, data_valid, = train_test_split(data_train_valid, test_size=0.25, random_state=1)

        return data_train, data_valid, data_test

    def _compute_fasttext_embeddings(self):
        print("Computing FastText embeddings...")

        # Load and tokeniz the corpus
        token_tokenized_corpus = self._load_tokenized_corpus()

        # Params
        enc_model = self.configs['models']['encoder']['model']
        emb_dim = self.configs['models']['encoder'][enc_model]['emb_dim']
        # Taken from Lample
        min_count = 0
        window_size = 5
        negative = 10
        n_epochs = 10

        # Train model
        print("Training FastText model...")
        ft_model = FastText(token_tokenized_corpus,
                            vector_size=emb_dim,
                            window=window_size,
                            min_count=min_count,
                            negative=negative,
                            sg=1, # SkipGram
                            epochs=n_epochs,
                            callbacks=[callback(n_epochs = n_epochs)])

        # Get embedding matrix
        embed_layer = nn.Embedding(self.configs['dataset']['vocab_size'], 
                                   emb_dim, 
                                   padding_idx = self.configs['dataset']['pad_index'])
        matrix_embedd = embed_layer.weight.detach() # Not that as this a pointer, changing matrix_embed will change embed_layer weights
        wv = ft_model.wv
        key_to_index = wv.key_to_index
        print("Computing embedding for {} tokens over {}".format(
            len(key_to_index),
            self.configs['dataset']['vocab_size']
        ))
        #index_to_key = {v:k for k,v in key_to_index.items()}
        ignored_tokens = 0 # Token that doen't have embedding
        for k in range(self.configs['dataset']['vocab_size']):
            token = self.tokenizer.id_to_token(k)
            if token in key_to_index.keys():
                # Token has an embedding
                matrix_embedd[k] = torch.tensor(wv[token])
            else:
                ignored_tokens += 1

        print("Ignored tokens : %s " % ignored_tokens)

        # Saving
        torch.save(embed_layer.state_dict(), self.embed_layer_name)
        print("Embedding layer saved at {}".format(self.embed_layer_name))

    def _load_tokenized_corpus(self):
        print("Tokenizing corpus...")
        token_tokenized_corpus = []
        for k in range(len(self.data)):
            token_tokenized_corpus.append(
                self.tokenizer.encode(self.data[k][0]).tokens
            )
            token_tokenized_corpus.append(
                self.tokenizer.encode(self.data[k][1]).tokens
            )
        return token_tokenized_corpus

    @staticmethod
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalizeString(s):
        s = SimpleEN_FR.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    @staticmethod
    def filterPair(p):
        return  p[0].startswith(SimpleEN_FR.ENG_PREFIXES)

    @staticmethod
    def filterPairs(pairs):
        return [pair for pair in pairs if SimpleEN_FR.filterPair(pair)]

    def load_data(self):
        lines = open('data\\fr_en_simple\\%s_%s.txt' % (self.L1_name, self.L2_name), encoding='utf-8').\
                     read().strip().split('\n')
        pairs = [[self.normalizeString(s) for s in l.split('\t')[:2]] for l in lines]
        return SimpleEN_FR.filterPairs(pairs)


class ContinuousSquares(Languages):
    """
        First Toy languages.

        It consists in two squares L1 and L2.
    """

    def __init__(self, configs):
        super(ContinuousSquares, self).__init__(configs)
        
        # Generate sentences
        self.L1_data = self._create_data(1)
        self.L2_data = self._create_data(2)


    def _create_data(self, language: int) -> torch.Tensor:
        """
            Create sentences for language i.

            Sample is divided in 4 squares in which the
            sampling is uniform but the number of points
            sampled varies depending on the square. Thus
            the overall sampling is not uniform.
        """
        assert language == 1 or language == 2
        assert self.configs['dataset']['L{}'.format(language)]['probs']['top_left'] +\
               self.configs['dataset']['L{}'.format(language)]['probs']['bottom_left'] +\
               self.configs['dataset']['L{}'.format(language)]['probs']['top_right'] +\
               self.configs['dataset']['L{}'.format(language)]['probs']['bottom_right'] - 1. < 1e-2

        # number of points to sample in each squares
        n_tl = int(self.configs['dataset']['size'] *\
                   self.configs['dataset']['L{}'.format(language)]['probs']['top_left'])
        n_bl = int(self.configs['dataset']['size'] *\
                   self.configs['dataset']['L{}'.format(language)]['probs']['bottom_left'])
        n_tr = int(self.configs['dataset']['size'] *\
                   self.configs['dataset']['L{}'.format(language)]['probs']['top_right'])
        n_br = int(self.configs['dataset']['size'] *\
                   self.configs['dataset']['L{}'.format(language)]['probs']['bottom_right'])

        # length of the squares
        lx = self.configs['dataset']['L{}'.format(language)]['positions']['x_max'] -\
             self.configs['dataset']['L{}'.format(language)]['positions']['x_min']
        ly = self.configs['dataset']['L{}'.format(language)]['positions']['y_max'] -\
             self.configs['dataset']['L{}'.format(language)]['positions']['y_min']

        # Sample uniformly points in each sub squares 
        sentences_tl = torch.tensor(
                            np.random.uniform(
                                low = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_min'], 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_min'] + ly/2
                                    ),
                                high = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_min'] + lx/2, 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_max']
                                    ), 
                                size = (
                                    n_tl, 
                                    self.configs['dataset']['L']['dim'])
                                ), 
                            dtype = torch.float32)
        sentences_bl = torch.tensor(
                            np.random.uniform(
                                low = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_min'], 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_min']
                                    ),
                                high = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_min'] + lx/2, 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_min'] + ly/2
                                    ), 
                                size = (
                                    n_bl, 
                                    self.configs['dataset']['L']['dim'])
                                ), 
                            dtype = torch.float32)
        sentences_tr = torch.tensor(
                            np.random.uniform(
                                low = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_min'] + lx/2, 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_min'] + ly/2
                                    ),
                                high = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_max'], 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_max']
                                    ), 
                                size = (
                                    n_tr, 
                                    self.configs['dataset']['L']['dim'])
                                ), 
                            dtype = torch.float32)
        sentences_br = torch.tensor(
                            np.random.uniform(
                                low = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_min'] + lx/2, 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_min']
                                    ),
                                high = (
                                    self.configs['dataset']['L{}'.format(language)]['positions']['x_max'], 
                                    self.configs['dataset']['L{}'.format(language)]['positions']['y_min'] + ly/2
                                    ), 
                                size = (
                                    n_br, 
                                    self.configs['dataset']['L']['dim'])
                                ), 
                            dtype = torch.float32)

        return torch.vstack([sentences_tl,
                             sentences_bl,
                             sentences_tr,
                             sentences_br])


class GaussianMixture(Languages):
    """
        Second Toy languages.

        It consists in a gaussian mixture
    """

    def __init__(self, configs):
        super(GaussianMixture, self).__init__(configs)
        
        # Generate sentences
        self.L1_data = self._create_data(1)
        self.L2_data = self._create_data(2)


    def _create_data(self, language: int) -> torch.Tensor:
        """
            Create sentences for language i.

        """
        assert language == 1 or language == 2

        # means and sigmas
        mus = self.configs['dataset']['L']['mus{}'.format(language)]
        sigmas = self.configs['dataset']['L']['sigmas']

        # size
        size = self.configs['dataset']['size']
        
        sentences = []
        for _ in range(size):
            k = np.random.choice(len(mus))
            sentences.append(
                torch.tensor(
                    np.random.normal(mus[k], sigmas[k], 2),
                    dtype = torch.float32
                    )
            )

        return torch.stack(sentences)




    

        
        
