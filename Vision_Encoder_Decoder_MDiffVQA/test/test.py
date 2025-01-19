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
sys.path.append('/home/ljmarten/RadiologyQA/')
sys.path.append('/home/ljmarten/RadiologyQA/pycocoevalcap/')

from mymodels.swinbertcross import SwinBERTFinetuned

from mydatasets.mimic_dataset import mimic_Dataset
from pycocoevalcap.metrics import Evaluator

torch.set_float32_matmul_precision('medium')

####################################################################
# Load Arguments
####################################################################

parser = argparse.ArgumentParser(description='Train NLL for RRG.')

# Agrega argumentos posibles
parser.add_argument('--exp_name', type=str, help='Experiment name.')
parser.add_argument('--model_arch', type=str, help='Architecture to train')
parser.add_argument('--load_weights', type=str, default=None, help='Load weights.')

# Parsea los argumentos
args = parser.parse_args()

# Print de los valores de los argumentos
print(20*'*')
print('exp_name:', args.exp_name)
print('model_arch:', args.model_arch)
if args.load_weights != None:
    print("load_weights: ", args.load_weights)
    args.load_weights = "/home/ljmarten/RadiologyQA/EXPERIMENTS/" + args.load_weights
print(30*'*')

EXP_DIR_PATH = "/home/ljmarten/RadiologyQA/EXPERIMENTS/" + args.exp_name
if not os.path.exists(EXP_DIR_PATH):
    os.makedirs(EXP_DIR_PATH)

####################################################################
# Load Model
####################################################################

DICT_MODELS = {
    "SwinBERTFinetuned": SwinBERTFinetuned(),
}
device = 'cuda:0'
model = DICT_MODELS[args.model_arch]

# Freeze encoder wights
# for param in model.encoder.parameters():
#     param.requires_grad = False

if args.load_weights != None:
    model.load_state_dict(torch.load(args.load_weights),)
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

####################################################################
# DataLoader Class
####################################################################

batch_size = 8 # 16
accumulate_grad_batches = 8 # 8
num_workers = multiprocessing.cpu_count() - 1
print("Num workers", num_workers)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size,
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=test_dataset.get_collate_fn()
)

####################################################################
# Training settings
####################################################################

# Training hyperparameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))

####################################################################
# Testing
####################################################################

# Load model in GPU
model.to(device)
model.eval()

def save_results_to_csv(main_image_paths, ref_image_paths, questions, refs, hyps, exp_dirpath):
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
        "main_image_paths": main_image_paths,
        "ref_image_paths": ref_image_paths,
        "questions": questions,
        "references": refs,
        "hypotheses": hyps
    })

    # Specify the output file name
    output_file = os.path.join(exp_dirpath, 'results.csv')

    # Save the DataFrame to a CSV file
    output_data.to_csv(output_file, index=False)

print("\n---- Start Testing ----")
# Test
main_image_paths, ref_image_paths = [], []
questions = []
gts, res = [], []
with torch.no_grad():
    with tqdm(iter(test_dataloader), desc="Test", unit="batch") as tepoch:
        for steps, batch in enumerate(tepoch):
            pixel_values = batch['images'].to(device)
            questions_ids = batch['questions_ids'].to(device)
            questions_mask  = batch['questions_mask'].to(device)
            answers_ids = batch['answers_ids'].to(device)
            answers_mask  = batch['answers_mask'].to(device)
            images_mask  = batch['images_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # decoder_out = model(questions_ids=questions_ids, 
            #                     questions_mask=questions_mask,
            #                     answers_ids=answers_ids, 
            #                     answers_mask=answers_mask,
            #                     images=pixel_values, 
            #                     images_mask=images_mask,
            #                     labels=labels)   
                    
            # loss = decoder_out['loss']

            generated_answers, _ = model.generate(
                pixel_values, images_mask=images_mask,
                questions_ids=questions_ids,
                questions_mask=questions_mask,
                tokenizer=model.tokenizer,
                num_beams=2,
                max_len=128,
                return_dict_in_generate=True,
                output_scores=True)

            for mip, rip, q, ra, ga in zip(batch['main_image_paths'], batch['ref_image_paths'], batch['questions'], batch['answers'], generated_answers):
                main_image_paths.append(mip)
                ref_image_paths.append(rip)
                questions.append(q)
                gts.append(ra)
                res.append(ga)

            # statistics
            # test_loss += loss.item()

            # tepoch.set_description(f'Metrics for Test | Loss: {loss.item():.4f}')

# Save results to CSV
save_results_to_csv(main_image_paths, ref_image_paths, questions, gts, res, EXP_DIR_PATH)

gts = {k: [{ 'caption': v }] for k, v in enumerate(gts)}
res = {k: [{ 'caption': v }] for k, v in enumerate(res)}

print("\n\nCalculating Metrics...\n")
evaluator = Evaluator()
evaluator.do_the_thing(gts, res)

print()
for k, v in evaluator.evaluation_report.items():
    print(f"{k}: {v:.3f}")
