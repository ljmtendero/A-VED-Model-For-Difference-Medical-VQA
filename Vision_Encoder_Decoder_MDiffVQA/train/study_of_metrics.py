import pandas as pd
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"

from myscorers.meteor.meteor import MeteorScorer

# Meteor
meteor_scorer = MeteorScorer()

# Load the data
df = pd.read_csv('/home/maasala/RadiologyQA/EXPERIMENTS/EfficientNet/results_epoch_9.csv')

# Calculate the Meteor score for each row
with open('/home/maasala/RadiologyQA/EXPERIMENTS/EfficientNet/meteor_values.txt', 'w') as f:
    with tqdm(iter(df.iterrows()), total=len(df)) as pbar:
        for i, (ref, hyp) in pbar:
            meteor = meteor_scorer.compute_score(predictions=[hyp], references=[ref])
            f.write(f'{ref}\t{hyp}\t{meteor}\n')