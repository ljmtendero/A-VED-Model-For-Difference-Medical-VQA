import os
import sys
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"


# test_meteor.py
# from myscorers.meteor.meteor import MeteorScorer
# from myscorers.bertscore.bertscore import BertScorer
from myscorers.chexbert.chexbert import myF1ChexBert

# Initialize the scorer
# meteor_scorer = MeteorScorer()
# bert_scorer = BertScorer()
f1_chexbert_scorer = myF1ChexBert()

# Example predictions and references
references = [
    # "the main image has an additional finding of consolidation than the reference image",
    "nothing has changed .",
    # "the main image has an additional finding of hilar congestion than the reference image . the main image is missing the findings of pleural effusion , edema , atelectasis , lung opacity , pneumonia , and cardiomegaly than the reference image ."
]

predictions = [
    # "the main image has an additional finding of lung opacity than the reference image",
    # "the main image is missing the finding of consolidation than the reference image",
    "nothing has changed .",
    # "the main image is missing the finding of edema than the reference image ."
]

# Compute METEOR score
# meteor_score = meteor_scorer.compute_score(predictions=predictions, references=references)

# BertScore
# bert_score = bert_scorer(predictions, references)

# F1 ChexBert
f1_chexbert = f1_chexbert_scorer.calculate(references, predictions)

# Print the result
# print("METEOR Score:", meteor_score)
# print("BertScore:", bert_score)
print("F1 ChexBert:", f1_chexbert)
