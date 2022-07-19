
from datasets import load_metric
import torch.nn as nn



class LanguageSim:
    def __init__(self, language_similarity: str = 'bleu',
                       tokenizer = None):

        self.language_similarity = language_similarity

        self.tokenizer = tokenizer

        if language_similarity == 'bleu':
            # As we need to store batches we need a metric by language
            self.bleu_metric_1 = load_metric('bleu') 
            self.bleu_metric_2 = load_metric('bleu')
        elif language_similarity == 'mse':
            self.mse_metric = nn.MSELoss()
        else:
            raise Exception(
                    'This language similarity measure is not implemented!'.format(
                                                            language_similarity
                                                            )
                            )

    def compute(self, preds = None, targets = None, language: int = 1):
        if self.language_similarity == 'bleu':
            if language == 1:
                return self.bleu_metric_1.compute()['bleu']
            elif language == 2:
                return self.bleu_metric_2.compute()['bleu']

        elif self.language_similarity == 'mse':

            return self.mse_metric(
                        preds, 
                        targets
                        ).item()

        elif self.language_similarity == 'cross_entropy':

            return self.cross_entropy_metric(
                        preds, 
                        targets
                        ).item()

    def add_batch(self, preds, targets, language: int = 1):
        if self.language_similarity == 'bleu':
            
            preds_text = self.tokenizer.decode_batch(preds.cpu().tolist())
            preds_text = [s.replace('@@', '').split(' ') for s in preds_text]

            targets_text = self.tokenizer.decode_batch(targets.cpu().tolist())
            targets_text = [[s.replace('@@', '').split(' ')] for s in targets_text]

            if language == 1:
                self.bleu_metric_1.add_batch(
                            predictions = preds_text,
                            references = targets_text
                            )
            elif language == 2:
                self.bleu_metric_2.add_batch(
                            predictions = preds_text,
                            references = targets_text
                            )
