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


from mydatasets.mydatasets_utils import ifcc_clean_report  # Import the preprocessing function

# Define a sample text
sample_text = """
This is a radiology report. It contains findings about the chest X-ray.
There are no significant abnormalities.
"""

# Specify the preprocessing function name as a string
text_preprocessing_function_name = "ifcc_clean_report"

# Use eval to dynamically get the function
text_preprocessing = eval(text_preprocessing_function_name)

# Apply the function to the sample text
processed_text = text_preprocessing(sample_text)

# Print the original and processed text
print("Original Text:")
print(sample_text)
print("\nProcessed Text:")
print(processed_text)
