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
from myscorers.cider.cider import Cider

# Example ground-truth references and generated captions
gts = {
    "image1": ["A man is playing a guitar.", "A person plays an acoustic guitar."],
    "image2": ["A cat sits on a mat.", "There is a cat on the mat."]
}

res = {
    "image1": ["A person playing a guitar."],
    "image2": ["A cat on the mat."]
}

# Initialize CIDEr scorer
cider_scorer = Cider(n=4, sigma=6.0)

# Compute CIDEr score
score, scores = cider_scorer.compute_score(gts=gts, res=res)

# Output results
print("CIDEr Corpus Score:", score)  # Overall CIDEr score
print("CIDEr Scores Per Image:", scores)  # Scores for each image
