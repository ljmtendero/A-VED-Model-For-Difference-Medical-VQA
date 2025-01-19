import warnings
warnings.filterwarnings("ignore")

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

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"

from mymodels.swinbert9k import SwinBERT9k
from mymodels.resswinbert import ResidualSwinBert
from mymodels.ressefficientbert import ResidualEfficientBert
# from mymodels.tinyvitbert import TinyViTBert
from mymodels.effnetv2sbert import EffNetv2sBert
from mymodels.effnetbert import EffNetBert
from mymodels.ressefficientbertconv import ResidualEfficientBertConv
from mymodels.effnetattbert import EffNetAttBert

from mydatasets.mimic_dataset import mimic_Dataset
from train.metrics import metrics_to_log

torch.set_float32_matmul_precision('medium')

####################################################################
# Load Arguments
####################################################################

parser = argparse.ArgumentParser(description='Train NLL for RRG.')

# Agrega argumentos posibles
parser.add_argument('--exp_name', type=str, help='Experiment name.')
parser.add_argument('--model_arch', type=str, help='Architecture to train')
parser.add_argument('--load_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--train_set', type=str, default="train", help='Load weights.')
parser.add_argument('--metrics_on_train', action='store_true', default=False, help='Calculate metrics on train.')

# Parsea los argumentos
args = parser.parse_args()

# Print de los valores de los argumentos
print(30*'*')
print('exp_name:', args.exp_name)
print('model_arch:', args.model_arch)
assert args.load_weights, "Load weights must be specified"
print("load_weights: ", args.load_weights)
print(30*'*')

EXP_DIR_PATH = "../EXPERIMENTS/inference/" + args.exp_name
if not os.path.exists(EXP_DIR_PATH):
    os.makedirs(EXP_DIR_PATH)

####################################################################
# Load Model
####################################################################

DICT_MODELS = {
    "SwinBERT9k": SwinBERT9k(),
    "ResidualSwin": ResidualSwinBert(),
    "ResidualEfficient": ResidualEfficientBert(),
    # "TinyViTBert": TinyViTBert(),
    "EffNetv2sBert": EffNetv2sBert(),
    "EffNetBert": EffNetBert(),
    "EffNetBertConv":ResidualEfficientBertConv(),
    "EffNetAttBert": EffNetAttBert()
}
device = 'cuda:0'
model = DICT_MODELS[args.model_arch]

model.load_state_dict(torch.load(args.load_weights))
print("Model initialized with weights: ", args.load_weights, "!")

####################################################################
# Dataset Class
####################################################################

test_dataset = mimic_Dataset(
                transform=model.val_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "test",
                multi_image=2
                )

train_dataset = mimic_Dataset(
                transform=model.train_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = args.train_set,
                multi_image=2
                )

####################################################################
# DataLoader Class
####################################################################

batch_size = 64 # 16
accumulate_grad_batches = 1 # 8
num_workers = multiprocessing.cpu_count() - 1
print("Num workers", num_workers)
train_dataloader = DataLoader(
    train_dataset,
    batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=train_dataset.get_collate_fn())

test_dataloader = DataLoader(
    test_dataset, 
    batch_size,
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=test_dataset.get_collate_fn())

####################################################################
# Training settings
####################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))

####################################################################
# Training
####################################################################

# Load model in GPU
model.to(device)

def save_results_to_csv(l_refs, l_hyps, exp_dirpath, split):
    """
    Saves references and hypotheses to a CSV file in the experiment directory for a specific epoch.

    Parameters:
        l_refs (list): List of references.
        l_hyps (list): List of hypotheses.
        epoch (int): Epoch number.
        exp_dirpath (str): Path to the experiment directory.

    Returns:
        str: The file name of the saved CSV.
    """
    # Create a DataFrame from the lists
    output_data = pd.DataFrame({
        "references": l_refs,
        "hypotheses": l_hyps
    })

    # Specify the output file name
    output_file = os.path.join(exp_dirpath, f"results_{split}.csv")

    # Save the DataFrame to a CSV file
    output_data.to_csv(output_file, index=False)

    return output_file

print("\n---- Start Inference ----")
train_loss, test_loss = 0., 0.
test_refs, test_hyps = [], []
train_refs, train_hyps = [], []
model.eval()
with torch.no_grad():
    if args.metrics_on_train:
        with tqdm(iter(train_dataloader), desc="Metrics for train", unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                questions_ids = batch['questions_ids'].to(device)
                questions_mask  = batch['questions_mask'].to(device)
                answers_ids = batch['answers_ids'].to(device)
                answers_mask  = batch['answers_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)

                decoder_out = model(questions_ids=questions_ids, 
                                    questions_mask=questions_mask,
                                    answers_ids=answers_ids, 
                                    answers_mask=answers_mask,
                                    images=pixel_values, 
                                    images_mask=images_mask)        
                loss = decoder_out['loss']

                generated_answers, _ = model.generate(
                    pixel_values, images_mask=images_mask,
                    questions_ids=questions_ids,
                    questions_mask=questions_mask,
                    tokenizer=model.tokenizer,
                    num_beams=2,
                    max_len=128,
                    return_dict_in_generate=True,
                    output_scores=True)

                reference_answers = batch['answers']

                for r, h in zip(reference_answers, generated_answers):
                    train_refs.append(r)
                    train_hyps.append(h)

                # statistics
                train_loss += loss.item()

                tepoch.set_description(f'Metrics for train - Loss: {loss.item():.4f}')

    with tqdm(iter(test_dataloader), desc="Metrics for test", unit="batch") as tepoch:
        for steps, batch in enumerate(tepoch):
            pixel_values = batch['images'].to(device)
            questions_ids = batch['questions_ids'].to(device)
            questions_mask  = batch['questions_mask'].to(device)
            answers_ids = batch['answers_ids'].to(device)
            answers_mask  = batch['answers_mask'].to(device)
            images_mask  = batch['images_mask'].to(device)

            decoder_out = model(questions_ids=questions_ids, 
                                questions_mask=questions_mask,
                                answers_ids=answers_ids, 
                                answers_mask=answers_mask,
                                images=pixel_values, 
                                images_mask=images_mask)        
            loss = decoder_out['loss']

            generated_answers, _ = model.generate(
                pixel_values, images_mask=images_mask,
                questions_ids=questions_ids,
                questions_mask=questions_mask,
                tokenizer=model.tokenizer,
                num_beams=2,
                max_len=128,
                return_dict_in_generate=True,
                output_scores=True)

            reference_answers = batch['answers']

            for r, h in zip(reference_answers, generated_answers):
                test_refs.append(r)
                test_hyps.append(h)

            # statistics
            test_loss += loss.item()

            tepoch.set_description(f'Metrics for Test - Loss: {loss.item():.4f}')

    if args.metrics_on_train:
        train_loss /= (len(train_dataloader.dataset) // batch_size)
        print(f'Train Loss: {train_loss:.4f}\n', end='')

    test_loss /= (len(test_dataloader.dataset) // batch_size)
    print(f'Test Loss: {test_loss:.4f}\n', end='')

    if args.metrics_on_train:
        file_name = save_results_to_csv(train_refs, train_hyps, EXP_DIR_PATH, "train")
        print(f"\nResultados de train guardados en: {file_name}", end='')

    file_name = save_results_to_csv(test_refs, test_hyps, EXP_DIR_PATH, "test")
    print(f"\nResultados de test guardados en: {file_name}\n")

    # Calculate metrics
    metrics_table, test_metrics = metrics_to_log(train_refs, train_hyps, test_refs, test_hyps, train_loss, test_loss)

    # Open the file in write mode ('w')
    with open(EXP_DIR_PATH + "/log.txt", 'w') as file:
        # Write the string to the file
        file.write(metrics_table + '\n')
    print(f'Metrics saved in {EXP_DIR_PATH}/log.txt')
