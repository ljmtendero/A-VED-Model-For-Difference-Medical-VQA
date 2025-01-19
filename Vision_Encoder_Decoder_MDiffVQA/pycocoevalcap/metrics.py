import pandas as pd
import numpy as np

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from tokenizer.ptbtokenizer import PTBTokenizer


class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score

    @staticmethod
    def metrics_to_log(evaluation_report, train_loss, test_loss, hnm_loss):
        df = pd.DataFrame(columns=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'RougeL', 'CIDEr', 'TR_Loss', 'TS_Loss', 'HNM_Loss'])
        df.loc[0] = [np.nan] * 10
        df.loc[1] = [np.nan] * 7 + [train_loss, test_loss, hnm_loss]
        df.index = ['Test', '']

        df.loc['Test', 'BLEU1'] = evaluation_report['Bleu_1']
        df.loc['Test', 'BLEU2'] = evaluation_report['Bleu_2']
        df.loc['Test', 'BLEU3'] = evaluation_report['Bleu_3']
        df.loc['Test', 'BLEU4'] = evaluation_report['Bleu_4']
        df.loc['Test', 'METEOR'] = evaluation_report['METEOR']
        df.loc['Test', 'RougeL'] = evaluation_report['ROUGE_L']
        df.loc['Test', 'CIDEr'] = evaluation_report['CIDEr']

        return df.round(4).fillna('').to_markdown(tablefmt='grid')

