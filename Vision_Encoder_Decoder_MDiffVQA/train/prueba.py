import os
import sys
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader
import json #QUITAR
import pandas as pd

sequences = torch.tensor([[147,  79,  75, 134, 113,  49,   6,   0, 129,   3]])


print(sequences)

sampled_ids = torch.stack([
    row[(row == 0).nonzero(as_tuple=True)[0][0] + 1:].contiguous() if (row == 0).any() else row.contiguous()
    for row in sequences
])


print(sampled_ids)

