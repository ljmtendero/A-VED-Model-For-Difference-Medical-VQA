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
from einops import rearrange

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"
sys.path.append('/home/ljmarten/RadiologyQA/')

from mymodels.swinbert9k import SwinBERT9k
from mymodels.resswinbert import ResidualSwinBert
from mymodels.ressefficientbert import ResidualEfficientBert
# from mymodels.tinyvitbert import TinyViTBert
from mymodels.effnetv2sbert import EffNetv2sBert
from mymodels.effnetbert import EffNetBert
from mymodels.ressefficientbertconv import ResidualEfficientBertConv
from mymodels.effnetattbert import EffNetAttBert
from mymodels.effnetbert_with_classification import EffNetBertWithClassification

from mydatasets.mimic_dataset import mimic_Dataset
from train.train_utils import multiassign, Hard_Negative_Mining
from train.metrics import metrics_to_log
from info_printer import *

torch.set_float32_matmul_precision('medium')

####################################################################
# Load Arguments
####################################################################

parser = argparse.ArgumentParser(description='Train NLL for RRG.')

# Agrega argumentos posibles
parser.add_argument('--exp_name', type=str, help='Experiment name.')
parser.add_argument('--model_arch', type=str, help='Architecture to train')
parser.add_argument('--load_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--hnm', type=bool, default=False, help='Use Hard Negative Mining.')
parser.add_argument('--train_set', type=str, default="train", help='Load weights.')
parser.add_argument('--metrics_on_train', action='store_true', default=False, help='Calculate metrics on train.')

# Parsea los argumentos
args = parser.parse_args()

# args.exp_name = 'NLL_BCE_EffNetBert'
# args.model_arch = 'EffNetBert_With_Classification'
args.hnm = True # False

# Print de los valores de los argumentos
print(20*'*')
print('exp_name:', args.exp_name)
print('model_arch:', args.model_arch)
if args.load_weights != None:
    print("load_weights: ", args.load_weights)
    args.load_weights = "../EXPERIMENTS/" + args.load_weights
print('hnm:', args.hnm)
print('train_set:', args.train_set)
print(30*'*')

EXP_DIR_PATH = "/home/ljmarten/RadiologyQA/EXPERIMENTS/" + args.exp_name
if not os.path.exists(EXP_DIR_PATH):
    os.makedirs(EXP_DIR_PATH)

####################################################################
# Load Model
####################################################################

DICT_MODELS = {
    # "SwinBERT9k": SwinBERT9k(),
    # "ResidualSwin": ResidualSwinBert(),
    # "ResidualEfficient": ResidualEfficientBert(),
    # "TinyViTBert": TinyViTBert(),
    # "EffNetv2sBert": EffNetv2sBert(),
    # "EffNetBert": EffNetBert(),
    # "EffNetBertConv":ResidualEfficientBertConv(),
    # "EffNetAttBert": EffNetAttBert(),
    "EffNetBert_With_Classification": EffNetBertWithClassification()
}
device = 'cuda:0'
model = DICT_MODELS[args.model_arch]

if args.load_weights != None:
    model.load_state_dict(torch.load(args.load_weights))
    print("Model initialized with weights: ", args.load_weights, "!")

####################################################################
# Dataset Class
####################################################################
multi_image = 2
test_dataset = mimic_Dataset(
    transform=model.val_transform, 
    tokenizer=model.tokenizer,
    processor=model.processor,
    partition = "test",
    multi_image=multi_image,
)

train_dataset = mimic_Dataset(
    transform=model.train_transform, 
    tokenizer=model.tokenizer,
    processor=model.processor,
    partition = args.train_set,
    multi_image=multi_image,
)

####################################################################
# DataLoader Class
####################################################################

batch_size = 32 # 16
accumulate_grad_batches = 4 # 8
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

# Training hyperparameters
epochs=30
criterion = nn.NLLLoss()
bce_loss_func = nn.BCEWithLogitsLoss()
#optimizer = optim.AdamW(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8) 
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))

####################################################################
# Save Training Information
####################################################################

with open(EXP_DIR_PATH + "/train_info.txt", "w") as outfile:
    outfile.write(print_soft_hard_info() + "\n")
    outfile.write(print_train_info(batch_size, epochs, num_workers) + "\n")
    outfile.write(print_model_info(model) + "\n")
    outfile.write(print_hyperparams_info(criterion, optimizer, lr_scheduler) + "\n")

####################################################################
# Training
####################################################################

# Load model in GPU
model.to(device)

#model = torch.compile(model, mode="reduce-overhead")
#model = torch.compile(model)


def save_results_to_csv(l_refs, l_hyps, epoch, exp_dirpath, split):
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
    output_file = os.path.join(exp_dirpath, f"results_{split}_epoch_{epoch}.csv")

    # Save the DataFrame to a CSV file
    output_data.to_csv(output_file, index=False)

    return output_file

best_bleu1 = float('-inf')
best_cider = float('-inf')
best_meteor = float('-inf')
best_bertscore = float('-inf')
best_f1_chexbert = float('-inf')
epoch_best_bleu1 = 0.
epoch_best_cider = 0.
epoch_best_meteor = 0.
epoch_best_bertscore = 0.
epoch_best_f1_chexbert = 0.

print('='*10, 'Training Started', '='*10, end='')
for epoch in range(epochs):
    # Train
    train_loss = 0.
    test_loss = 0.
    if args.hnm:
        train_hnm_loss = 0
        dict_loss = {}
    if args.metrics_on_train:
        metrics_for_train_loss = 0.
    model.train()
    optimizer.zero_grad()
    print(f'\nEpoch {epoch+1}/{epochs}')
    with tqdm(iter(train_dataloader), desc="Train", unit="batch") as tepoch:
        for steps, batch in enumerate(tepoch):
            ids = batch["idx"].to('cpu').numpy()
            images = batch['images'].to(device)
            images_masks = batch['images_masks'].to(device)
            questions = batch['questions'].to(device)
            questions_masks = batch['questions_masks'].to(device)
            answers = batch['answers'].to(device)
            answers_masks = batch['answers_masks'].to(device)
            labels = batch['labels'].to(device)
            images_labels = batch['images_labels'].to(device)

            # Visual Encoder
            images_features, images_features_masks, images_logits = model.encode(images, images_masks)
            images_labels = rearrange(images_labels, 'b s c -> (b s) c', s=multi_image)
            bce_loss = bce_loss_func(images_logits, images_labels)

            # Text Decoder
            nll_loss = model.decode(
                questions=questions,
                questions_masks=questions_masks,
                answers=answers,
                answers_masks=answers_masks,
                encoder_hidden_states=images_features,
                encoder_attention_mask=images_features_masks,
                labels=labels,)                
            loss = nll_loss + bce_loss

            # Calculate gradients
            loss.backward()

            if steps % accumulate_grad_batches == 0 and steps != 0:
                # Update parameters
                optimizer.step()
                # zero the parameter gradients
                optimizer.zero_grad()

            if args.hnm:
                multiassign(dict_loss, ids, 
                                [loss.to('cpu').detach().numpy()])
            # statistics
            train_loss += loss.item()
            tepoch.set_description(f'Train | Loss: {loss.item():.4f}')
        
        optimizer.zero_grad()

    # HNM
    if args.hnm:
        HNM_trainloader = Hard_Negative_Mining(
            dict_loss, train_dataset, batch_size, num_workers=num_workers)
        
        with tqdm(iter(HNM_trainloader), desc="HNM", unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                ids = batch["idx"].to('cpu').numpy()
                images = batch['images'].to(device)
                images_masks = batch['images_masks'].to(device)
                questions = batch['questions'].to(device)
                questions_masks = batch['questions_masks'].to(device)
                answers = batch['answers'].to(device)
                answers_masks = batch['answers_masks'].to(device)
                labels = batch['labels'].to(device)
                images_labels = batch['images_labels'].to(device)

                # Visual Encoder
                images_features, images_features_masks, images_logits = model.encode(images, images_masks)
                images_labels = rearrange(images_labels, 'b s c -> (b s) c', s=multi_image)
                bce_loss = bce_loss_func(images_logits, images_labels)

                # Text Decoder
                nll_loss = model.decode(
                    questions=questions,
                    questions_masks=questions_masks,
                    answers=answers,
                    answers_masks=answers_masks,
                    encoder_hidden_states=images_features,
                    encoder_attention_mask=images_features_masks,
                    labels=labels,)                
                loss = nll_loss + bce_loss

                # Calculate gradients
                loss.backward()

                if steps % accumulate_grad_batches == 0 and steps != 0:
                    # Update parameters
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                 # statistics
                train_hnm_loss += loss.item()

                #tepoch.set_description("loss: " % str(loss.item()))
                tepoch.set_description(f'HNM | Loss: {loss.item():.4f}')
            
            optimizer.zero_grad()
            
    #lr_scheduler.step()

    # Test
    test_refs, test_hyps = [], []
    train_refs, train_hyps = [], []
    model.eval()
    with torch.no_grad():
        if args.metrics_on_train:
            with tqdm(iter(train_dataloader), desc="Metrics for Train" + str(epoch), unit="batch") as tepoch:
                for steps, batch in enumerate(tepoch):
                    ids = batch["idx"].to('cpu').numpy()
                    images = batch['images'].to(device)
                    images_masks = batch['images_masks'].to(device)
                    questions = batch['questions'].to(device)
                    questions_masks = batch['questions_masks'].to(device)
                    answers = batch['answers'].to(device)
                    answers_masks = batch['answers_masks'].to(device)
                    labels = batch['labels'].to(device)
                    images_labels = batch['images_labels'].to(device)

                    # Visual Encoder
                    images_features, images_features_masks, images_logits = model.encode(images, images_masks)
                    # bce_loss = bce_loss_func(images_logits, images_labels)

                    # Text Decoder
                    # nll_loss = model.decode(
                    #     questions=questions,
                    #     questions_masks=questions_masks,
                    #     answers=answers,
                    #     answers_masks=answers_masks,
                    #     encoder_hidden_states=images_features,
                    #     encoder_attention_mask=images_features_masks,
                    #     labels=labels,)                
                    # loss = nll_loss + bce_loss

                    generated_answers, _ = model.generate(
                        input_ids=questions,
                        encoder_hidden_states=images_features,
                        encoder_attention_mask=images_features_masks,
                        tokenizer=model.tokenizer,
                        num_beams=2,
                        max_len=128,
                        return_dict_in_generate=True,
                        output_scores=True)

                    reference_answers = batch['raw_answers']

                    for r, h in zip(reference_answers, generated_answers):
                        train_refs.append(r)
                        train_hyps.append(h)

                    # tepoch.set_description(f'Metrics for Train | Loss: {loss.item():.4f}')
                    tepoch.set_description(f'Metrics for Train')

        with tqdm(iter(test_dataloader), desc="Metrics for Test" + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                ids = batch["idx"].to('cpu').numpy()
                images = batch['images'].to(device)
                images_masks = batch['images_masks'].to(device)
                questions = batch['questions'].to(device)
                questions_masks = batch['questions_masks'].to(device)
                answers = batch['answers'].to(device)
                answers_masks = batch['answers_masks'].to(device)
                labels = batch['labels'].to(device)
                images_labels = batch['images_labels'].to(device)

                # Visual Encoder
                images_features, images_features_masks, images_logits = model.encode(images, images_masks)
                images_labels = rearrange(images_labels, 'b s c -> (b s) c', s=multi_image)
                bce_loss = bce_loss_func(images_logits, images_labels)

                # Text Decoder
                nll_loss = model.decode(
                    questions=questions,
                    questions_masks=questions_masks,
                    answers=answers,
                    answers_masks=answers_masks,
                    encoder_hidden_states=images_features,
                    encoder_attention_mask=images_features_masks,
                    labels=labels,)                
                loss = nll_loss + bce_loss

                generated_answers, _ = model.generate(
                    input_ids=questions,
                    encoder_hidden_states=images_features,
                    encoder_attention_mask=images_features_masks,
                    tokenizer=model.tokenizer,
                    num_beams=2,
                    max_len=128,
                    return_dict_in_generate=True,
                    output_scores=True)

                reference_answers = batch['raw_answers']

                for r, h in zip(reference_answers, generated_answers):
                    test_refs.append(r)
                    test_hyps.append(h)

                # statistics
                test_loss += loss.item()

                tepoch.set_description(f'Metrics for Test | Loss: {loss.item():.4f}')
                # tepoch.set_description(f'Metrics for Test')

    train_loss /= (len(train_dataloader.dataset) // batch_size)
    test_loss /= (len(test_dataloader.dataset) // batch_size)
    print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}', end='')

    if args.hnm:
        train_hnm_loss /= (len(HNM_trainloader.dataset) // batch_size)
        print(f" | Train HNM Loss: {train_hnm_loss:.4f}")
    else:
        print("")

    if args.metrics_on_train:
        file_name = save_results_to_csv(train_refs, train_hyps, epoch, EXP_DIR_PATH, "train")
        print(f"\nResultados de train guardados en: {file_name}", end='')

    file_name = save_results_to_csv(test_refs, test_hyps, epoch, EXP_DIR_PATH, "test")
    print(f"\nResultados de test guardados en: {file_name}\n")

    # Calculate metrics
    metrics_table, test_metrics = metrics_to_log(train_refs, train_hyps, test_refs, test_hyps, train_loss, test_loss, train_hnm_loss)

    if best_bleu1 < test_metrics['BLEU1']:
        try:
            os.remove(EXP_DIR_PATH + "/best_bleu1_" + str(epoch_best_bleu1) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_bleu1 = test_metrics['BLEU1']
        epoch_best_bleu1 = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_bleu1_" + str(epoch) + "_model.pt")

    if best_cider < test_metrics['CIDEr']:
        try:
            os.remove(EXP_DIR_PATH + "/best_cider_" + str(epoch_best_cider) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_cider = test_metrics['CIDEr']
        epoch_best_cider = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_cider_" + str(epoch) + "_model.pt")

    if best_meteor < test_metrics['METEOR']:
        try:
            os.remove(EXP_DIR_PATH + "/best_meteor_" + str(epoch_best_meteor) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_meteor = test_metrics['METEOR']
        epoch_best_meteor = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_meteor_" + str(epoch) + "_model.pt")

    if best_bertscore < test_metrics['BertScore']:
        try:
            os.remove(EXP_DIR_PATH + "/best_bertscore_" + str(epoch_best_bertscore) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_bertscore = test_metrics['BertScore']
        epoch_best_bertscore = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_bertscore_" + str(epoch) + "_model.pt")

    if best_f1_chexbert < test_metrics['F1 ChexBert']:
        try:
            os.remove(EXP_DIR_PATH + "/best_f1_chexbert_" + str(epoch_best_f1_chexbert) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_f1_chexbert = test_metrics['F1 ChexBert']
        epoch_best_f1_chexbert = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_f1_chexbert_" + str(epoch) + "_model.pt")

    lr_scheduler.step(test_loss)

    EXP_DIR_PATH
    # Open the file in write mode ('w')
    with open(EXP_DIR_PATH + "/log.txt", 'a') as file:
        # Write the string to the file
        file.write("EPOCH: " + str(epoch) + "\n")
        file.write(metrics_table + "\n\n")
    print(f'Metrics saved in {EXP_DIR_PATH}/log.txt\n')

# Save Final weights
torch.save(model.state_dict(), EXP_DIR_PATH + "/last_model.pt")
