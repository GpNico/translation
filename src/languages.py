"""
    blabla
"""   
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SentencesDataset(Dataset):
    """
        Create Pytorch Dataset from given inputs.
            - sentences : [tensor] directly the sentences (toy examples) 
                                   or the tokenized sentences (real examples)
            - sentences_ids : [tensor] in the discrete stochastic version contains
                                       ids of either the whole sentence (toy 
                                       examples) either each tokens (real examples)

    """
    def __init__(self, sentences: torch.Tensor,
                       sentences_ids: torch.Tensor = None):
        self.sentences = sentences
        self.sentences_ids = sentences_ids

    def __len__(self) -> int:
        return self.sentences.shape[0]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        item_dict = {'sentences': self.sentences[item]}
        if self.sentences_ids != None:
            item_dict['sentences_ids'] = self.sentences_ids[item] 
        return item_dict


class Languages:
    """
        Parent Class for Languages.
    """

    def __init__(self, configs) -> None:
        self.configs = configs
        # Sentences (= datasets)
        self.L1_sentences = None
        self.L2_sentences = None

        # In sequence languages we need vocabulary
        self.L1_vocabulary = None # For sequence languages
        self.L2_vocabulary = None # rq: in case of joint tokenization L1_vocabulary = L2_vocabulary


    def _get_datasets(self, language: int) -> tuple[SentencesDataset]:
        """
            Construct datasets for train, Validation
            and test.
        """
        assert language == 1 or language == 2

        sentences_train, sentences_valid, sentences_test = self._train_valid_test_split(language)
        sentences_train_ids = self._create_sentences_ids(sentences_train)
        sentences_valid_ids = self._create_sentences_ids(sentences_valid)
        sentences_test_ids = self._create_sentences_ids(sentences_test)

        train_data = SentencesDataset(sentences_train,
                                      sentences_train_ids)
        valid_data = SentencesDataset(sentences_valid,
                                      sentences_valid_ids)
        test_data = SentencesDataset(sentences_test,
                                     sentences_test_ids)

        return train_data, valid_data, test_data

    def get_dataloaders(self, language: int) -> tuple[DataLoader]:
        """
            Construct dataloaders for train, validation
            and test. 
        """
        assert language == 1 or language == 2

        train_data, valid_data, test_data = self._get_datasets(language)

        train_loader = DataLoader(train_data, 
                                  batch_size = self.configs['batch_size'], 
                                  shuffle = True, 
                                  num_workers = 0, 
                                  drop_last = True)

        valid_loader = DataLoader(valid_data, 
                                  batch_size = self.configs['batch_size'], 
                                  shuffle = False, 
                                  num_workers = 0, 
                                  drop_last = True)

        test_loader = DataLoader(test_data, 
                                 batch_size = self.configs['batch_size'], 
                                 shuffle = False, 
                                 num_workers = 0, 
                                 drop_last = True)

        return train_loader, valid_loader, test_loader


    def _train_valid_test_split(self, language: int) -> tuple[torch.Tensor]:
        """
           Split sentences in three train, valiation
           and test sets respectively.
        """
        assert language == 1 or language == 2

        if language == 1:
            sentences = self.L1_sentences
        elif language == 2:
            sentences = self.L2_sentences

        sentences_train_valid, sentences_test = train_test_split(sentences, test_size=0.2, random_state=1)

        sentences_train, sentences_valid, = train_test_split(sentences_train_valid, test_size=0.25, random_state=1)

        return sentences_train, sentences_valid, sentences_test


    def _create_sentences(self, language: int) -> torch.Tensor:
        """
            Create sentences of the laznguage i.
            Specific to each languages.
            TBD
        """
        raise Exception("_create_sentences is not implemented!")

    def _create_sentences_ids(self, sentences: torch.Tensor) -> torch.Tensor:
        """
            Map the given sentences to their ids.
        """
        return None



class DiscreteSquares(Languages):
    """
        First Stochastic languages.

        It consists in a dicrete grid L and two squares L1 and L2
        included in L.

        To obtain a stochastic language one has to create synonyms,
        and it the DiscreteSquares setup synonyms are represented 
        on parallel plans :

        Syn( (x,y,0) ) = { (x,y,z) | z = k*eps for all - k_max <= k <= k_max } 

        where k is an integer, k_max and eps are language's parameters.
    """

    def __init__(self, configs):
        super(DiscreteSquares, self).__init__(configs)
        # Grid size
        self.Nx_L = int((configs['L_x_max'] - configs['L_x_min'])/configs['discrete_step']) + 1
        self.Ny_L = int((configs['L_y_max'] - configs['L_y_min'])/configs['discrete_step']) + 1
        self.Nz_L = 2*configs['max_k'] + 1

        # Parameter specific to this Languages
        self.key_factor = 1000
        assert configs['discrete_step']*self.key_factor > 1

        # Create ids for the grid
        self._mapping_one_hot()
        
        # Generate sentences
        self.L1_sentences = self._create_sentences(1)
        self.L2_sentences = self._create_sentences(2)


    def _create_sentences(self, language: int) -> torch.Tensor:
        """
            Create sentences for language i.

            Sample uniformly in the L_i square and
            then project on the grid define by L.

            Randomly offset some sentences in the 
            z dimension to create synonyms.
        """
        # Sample uniformly points in the L_i square 
        sentences = torch.tensor(
                            np.random.uniform(
                                low = (
                                    self.configs['L{}_x_min'.format(language)], 
                                    self.configs['L{}_y_min'.format(language)]
                                    ),
                                high = (
                                    self.configs['L{}_x_max'.format(language)], 
                                    self.configs['L{}_y_max'.format(language)]
                                    ), 
                                size = (
                                    self.configs['n_sentences'], 
                                    self.configs['dim_L']-1)
                                ), 
                            dtype = torch.float32)

        # Sample random offset to simulate synonyms
        offsets = torch.randint(low = -self.configs['max_k'],
                                high = self.configs['max_k'] + 1, 
                                size = (self.configs['n_sentences'],1)
                                )
        # Project the sentences on the grid and add offsets*eps
        sentences = torch.cat(
                        (sentences.div(self.configs['discrete_step'], 
                                       rounding_mode="floor") * self.configs['discrete_step'],
                         offsets*self.configs['epsilon{}'.format(language)]
                        ), 
                        dim = 1
                    )

        return sentences

    def _create_sentences_ids(self, sentences: torch.Tensor) -> torch.Tensor:
        """
            Map the given sentences to their ids.
            We use the - created for this purpose - 
            coords_to_id composed to _convert_coords_to_key.
            Args:
                sentences (tensor) [N, 3]
        """
        # Not beautiful (should probably use map method)
        n_sentences = sentences.shape[0]
        sentences_ids = []
        for k in range(n_sentences):
            sentence = sentences[k]
            key = self._convert_coords_to_key(sentence.tolist())
            sentences_ids.append(self.coords_to_id[key])
        return torch.tensor(sentences_ids, dtype = torch.long)

    def _convert_coords_to_key(self, coords: list[float]) -> str:
        """
            Compute the key associated to coordinates.
            Ex:
                coords = [0.5, 0.12, 0.6]
                -> key = '500_120_600'
        """
        x, y, z = coords
        return '{}_{}_{}'.format(int(x*self.key_factor),
                                 int(y*self.key_factor),
                                 int(z*self.key_factor))

    def _convert_key_to_coords(self, key: str) -> list[float]:
        """
            Compute the coordinates associated to a key.
            Ex:
                key : '500_120_600'
                -> coords = [0.5, 0.12, 0.6]
        """
        coords = key.split('_')
        return [float(x)/self.key_factor for x in coords]

    def _mapping_one_hot(self) -> None:
        """
            Map the grid to an id.
            Ex: x--x--x
                |  |  |
                x--x--x
                is mapped to :
                1--3--5
                |  |  |
                0--2--4
        """
        xs = np.linspace(self.configs['L_x_min'], self.configs['L_x_max'], self.Nx_L)
        ys = np.linspace(self.configs['L_y_min'], self.configs['L_y_max'], self.Ny_L)
        # We use min(eps1, eps2) to ensure that the grid is the same for
        # for L1 and L2
        eps = min(self.configs['epsilon1'], self.configs['epsilon2'])
        zs = np.array([k*eps for k in range(-self.configs['max_k'],
                                             self.configs['max_k'] + 1)])
        self.coords_to_id = {}
        self.id_to_coords = {}
        count = 0
        for x in xs:
            for y in ys:
                for z in zs:
                    key = self._convert_coords_to_key([x,y,z])
                    self.coords_to_id[key] = count
                    self.id_to_coords[count] = key
                    count += 1


    def convert_id_to_coords(self, ids: torch.Tensor) -> torch.Tensor:
        """
            Args:
                ids (tensor) shape [batch_size]
            Returns:
                coords (tensor) shape [batch_size, 3]
        """
        # Need to be optimized as it used in the BT
        # forward pass... 
        coords = []
        for pred_id in ids:
            key = self.id_to_coords[pred_id.item()]
            coords.append(self._convert_key_to_coords(key))
        return torch.tensor(coords)

class ContinuousSquares(Languages):
    """
        First Toy languages.

        It consists in two squares L1 and L2.
    """

    def __init__(self, configs):
        super(ContinuousSquares, self).__init__(configs)
        
        # Generate sentences
        self.L1_sentences = self._create_sentences(1)
        self.L2_sentences = self._create_sentences(2)


    def _create_sentences(self, language: int) -> torch.Tensor:
        """
            Create sentences for language i.

            Sample uniformly in the L_i square.
        """
        # Sample uniformly points in the L_i square 
        sentences = torch.tensor(
                            np.random.uniform(
                                low = (
                                    self.configs['L{}_x_min'.format(language)], 
                                    self.configs['L{}_y_min'.format(language)]
                                    ),
                                high = (
                                    self.configs['L{}_x_max'.format(language)], 
                                    self.configs['L{}_y_max'.format(language)]
                                    ), 
                                size = (
                                    self.configs['n_sentences'], 
                                    self.configs['dim_L'])
                                ), 
                            dtype = torch.float32)

        return sentences



    

        
        
