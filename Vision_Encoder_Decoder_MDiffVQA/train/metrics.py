import pandas as pd
pd.options.display.float_format = '{:.4f}'.format

import numpy as np
from typing import List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"

from myscorers.bleu.bleu import Bleu
from myscorers.meteor.meteor import MeteorScorer
from myscorers.cider.cider import Cider
from myscorers.rouge.rouge import Rouge
from myscorers.bertscore.bertscore import BertScorer
from myscorers.chexbert.chexbert import myF1ChexBert

bleu1_scorer = Bleu(n=1)
bleu2_scorer = Bleu(n=2)
bleu3_scorer = Bleu(n=3)
bleu4_scorer = Bleu(n=4)
meteor_scorer = MeteorScorer()
cider_scorer = Cider(n=4, sigma=6.0)
rougel_scorer = Rouge(rouges=['rougeL'])
bert_scorer = BertScorer()
f1_chexbert_scorer = myF1ChexBert()


def calculate_metrics(refs, hyps):
    # BLEU
    bleu1 = bleu1_scorer(refs, hyps)[0]
    bleu2 = bleu2_scorer(refs, hyps)[0]
    bleu3 = bleu3_scorer(refs, hyps)[0]
    bleu4 = bleu4_scorer(refs, hyps)[0]

    # METEOR
    meteor = meteor_scorer.compute_score(predictions=hyps, references=refs)

    # CIDEr
    gts = {i: [ref] for i, ref in enumerate(refs)}
    res = {i: [hyp] for i, hyp in enumerate(hyps)}
    cider, _ = cider_scorer.compute_score(gts=gts, res=res)

    # RougeL
    rougeL = rougel_scorer(refs=refs, hyps=hyps)[0]

    # BertScore
    bert_score = bert_scorer(hyps, refs)

    # F1 ChexBert
    f1_chexbert = f1_chexbert_scorer.calculate(refs, hyps)

    return {
        'BLEU1': bleu1,
        'BLEU2': bleu2,
        'BLEU3': bleu3,
        'BLEU4': bleu4,
        'METEOR': meteor,
        'CIDEr': cider,
        'RougeL': rougeL,
        'BertScore': bert_score,
        'F1 ChexBert': f1_chexbert
    }


def metrics_to_log(
        train_refs:List = [], 
        train_hyps:List = [], 
        test_refs:List = [], 
        test_hyps:List = [], 
        tr_loss:float = -1.0, 
        ts_loss:float = -1.0, 
        hnm_loss:float = -1.0):
    
    df = pd.DataFrame(columns=['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'CIDEr', 'RougeL', 'BertScore', 'F1 ChexBert', 'TR_Loss', 'TS_Loss', 'HNM_Loss'])
    df.loc[0] = [np.nan] * 12
    df.loc[1] = [np.nan] * 12
    df.loc[2] = [np.nan] * 9 + [tr_loss, ts_loss, hnm_loss]
    df.index = ['Train', 'Test', '']

    if train_refs and train_hyps:
        print('Calculating metrics for Train set...')
        df.loc['Train'] = calculate_metrics(train_refs, train_hyps)

    print('\nCalculating metrics for Test set...\n')
    test_metrics = calculate_metrics(test_refs, test_hyps)
    df.loc['Test'] = test_metrics

    return df.round(4).fillna('').to_markdown(tablefmt='grid'), test_metrics
