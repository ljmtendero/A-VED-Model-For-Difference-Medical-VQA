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
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"
sys.path.append('/home/ljmarten/RadiologyQA/')
sys.path.append('/home/ljmarten/RadiologyQA/pycocoevalcap/')

from mymodels.swinbertcross import SwinBERTFinetuned

from mydatasets.mimic_dataset import mimic_Dataset
from train.train_utils import multiassign, Hard_Negative_Mining
# from train.metrics import metrics_to_log
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
parser.add_argument('--hnm', type=bool, default=False, help='Use Hard Negative Mining.')
parser.add_argument('--train_set', type=str, default="train", help='Load weights.')
# parser.add_argument('--metrics_on_train', action='store_true', default=False, help='Calculate metrics on train.')

# Parsea los argumentos
args = parser.parse_args()

# Print de los valores de los argumentos
print(20*'*')
print('exp_name:', args.exp_name)
print('model_arch:', args.model_arch)
if args.load_weights != None:
    print("load_weights: ", args.load_weights)
    args.load_weights = "/home/ljmarten/RadiologyQA/EXPERIMENTS/" + args.load_weights
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

train_dataset = mimic_Dataset(
    transform=model.train_transform, 
    tokenizer=model.tokenizer,
    processor=model.processor,
    partition = args.train_set,
    multi_image=2
)

# print("Train dataset sample:")
# print(train_dataset[0])  # Print first sample of the dataset
# print("Test dataset sample:")
# print(test_dataset[0])  # Print first sample of the dataset

####################################################################
# DataLoader Class
####################################################################

batch_size = 8 # 16
accumulate_grad_batches = 8 # 8
num_workers = multiprocessing.cpu_count() - 1
print("Num workers", num_workers)
train_dataloader = DataLoader(
    train_dataset,
    batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=train_dataset.get_collate_fn()
)

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
epochs=30
criterion = nn.NLLLoss()
#optimizer = optim.AdamW(model.parameters(), lr=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.000003)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))

####################################################################
# Training
####################################################################

# Load model in GPU
model.to(device)

#model = torch.compile(model, mode="reduce-overhead")
#model = torch.compile(model)


def save_results_to_csv(main_image_paths=[], ref_image_paths=[], questions=[], refs=[], hyps=[], epoch=-1, exp_dirpath="", split=""):
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
    output_file = os.path.join(exp_dirpath, f"results_{split}_epoch_{epoch}.csv")

    # Save the DataFrame to a CSV file
    output_data.to_csv(output_file, index=False)

    return output_file

best_bleu1 = -9999999.9
best_cider = -9999999.9
best_meteor = -9999999.9
best_rougel = -9999999.9
best_bertscore = -9999999.9
best_f1_chexbert = -9999999.9
epoch_best_bleu1 = 0
epoch_best_cider = 0
epoch_best_meteor = 0
epoch_best_bertscore = 0
epoch_best_f1_chexbert = 0
epoch_best_rougel = 0

print("\n---- Start Training ----")
for epoch in range(epochs):

    # Train
    train_loss = 0
    test_loss = 0
    if args.hnm:
        train_hnm_loss = 0
        dict_loss = {}
    model.train()
    optimizer.zero_grad()
    with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
        for steps, batch in enumerate(tepoch):
            pixel_values = batch['images'].to(device)
            questions_ids = batch['questions_ids'].to(device)
            questions_mask  = batch['questions_mask'].to(device)
            answers_ids = batch['answers_ids'].to(device)
            answers_mask  = batch['answers_mask'].to(device)
            images_mask  = batch['images_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["idx"].to('cpu').numpy()

            decoder_out = model(questions_ids=questions_ids,
                                questions_mask=questions_mask,
                                answers_ids=answers_ids, 
                                answers_mask=answers_mask,
                                images=pixel_values, 
                                images_mask=images_mask,
                                labels=labels)
            # print("Decoder outputs before loss computation:", decoder_out)         
            loss = decoder_out['loss']

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

            #tepoch.set_description("loss: " % str(loss.item()))
            tepoch.set_description(f'Train Epoch [{epoch}/{epochs-1}] Loss: {loss.item():.4f}')

            #if steps == 4:
            #    break
        
        optimizer.zero_grad()

    # HNM
    if args.hnm:
        HNM_trainloader = Hard_Negative_Mining(
            dict_loss, train_dataset, batch_size, num_workers=num_workers)
        
        optimizer.zero_grad()
        
        with tqdm(iter(HNM_trainloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                questions_ids = batch['questions_ids'].to(device)
                questions_mask  = batch['questions_mask'].to(device)
                answers_ids = batch['answers_ids'].to(device)
                answers_mask  = batch['answers_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)
                labels = batch['labels'].to(device)

                decoder_out = model(questions_ids=questions_ids, 
                                    questions_mask=questions_mask,
                                    answers_ids=answers_ids, 
                                    answers_mask=answers_mask,
                                    images=pixel_values, 
                                    images_mask=images_mask,
                                    labels=labels)
                         
                loss = decoder_out['loss']

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
                tepoch.set_description(f'HNM Epoch [{epoch}/{epochs-1}] Loss: {loss.item():.4f}')

                #if steps == 4:
                #    break
            
            optimizer.zero_grad()
            
    #lr_scheduler.step()

    # Test
    main_image_paths, ref_image_paths = [], []
    questions = []
    test_refs, test_hyps = [], []
    # train_refs, train_hyps = [], []
    model.eval()
    with torch.no_grad():
        # if args.metrics_on_train:
        #     with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
        #         for steps, batch in enumerate(tepoch):
                    
        #             pixel_values = batch['images'].to(device)
        #             questions_ids = batch['questions_ids'].to(device)
        #             questions_mask  = batch['questions_mask'].to(device)
        #             answers_ids = batch['answers_ids'].to(device)
        #             answers_mask  = batch['answers_mask'].to(device)
        #             images_mask  = batch['images_mask'].to(device)
        #             labels = batch['labels'].to(device)

        #             decoder_out = model(questions_ids=questions_ids, 
        #                                 questions_mask=questions_mask,
        #                                 answers_ids=answers_ids, 
        #                                 answers_mask=answers_mask,
        #                                 images=pixel_values, 
        #                                 images_mask=images_mask,
        #                                 labels=labels)      
        #             loss = decoder_out['loss']

        #             generated_answers, _ = model.generate(
        #                 pixel_values, images_mask=images_mask,
        #                 questions_ids=questions_ids,
        #                 questions_mask=questions_mask,
        #                 tokenizer=model.tokenizer,
        #                 num_beams=2,
        #                 max_len=128,
        #                 return_dict_in_generate=True,
        #                 output_scores=True)

        #             reference_answers = batch['answers']

        #             for r, h in zip(reference_answers, generated_answers):
        #                 train_refs.append(r)
        #                 train_hyps.append(h)

        #             # statistics
        #             test_loss += loss.item()

        #             tepoch.set_description(f'Metrics for Train - Epoch [{epoch}/{epochs-1}] Loss: {loss.item():.4f}')

        with tqdm(iter(test_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                questions_ids = batch['questions_ids'].to(device)
                questions_mask  = batch['questions_mask'].to(device)
                answers_ids = batch['answers_ids'].to(device)
                answers_mask  = batch['answers_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)
                labels = batch['labels'].to(device)
                
                decoder_out = model(questions_ids=questions_ids, 
                                    questions_mask=questions_mask,
                                    answers_ids=answers_ids, 
                                    answers_mask=answers_mask,
                                    images=pixel_values, 
                                    images_mask=images_mask,
                                    labels=labels)   
                     
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

                for mip, rip, q, ra, ga in zip(batch['main_image_paths'], batch['ref_image_paths'], batch['questions'], batch['answers'], generated_answers):
                # for ra, ga in zip(reference_answers, generated_answers):
                    main_image_paths.append(mip)
                    ref_image_paths.append(rip)
                    questions.append(q)
                    test_refs.append(ra)
                    test_hyps.append(ga)

                # statistics
                test_loss += loss.item()

                tepoch.set_description(f'Metrics for Test - Epoch [{epoch}/{epochs-1}] Loss: {loss.item():.4f}')

    train_loss /= (len(train_dataloader.dataset) // batch_size)
    test_loss /= (len(test_dataloader.dataset) // batch_size)
    print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}', end='')

    if args.hnm:
        train_hnm_loss /= (len(HNM_trainloader.dataset) // batch_size)
        print(f" | Train HNM Loss: {train_hnm_loss}")
    else:
        print("")

    # if args.metrics_on_train:
    #     file_name = save_results_to_csv(train_refs, train_hyps, epoch, EXP_DIR_PATH, "train")
    #     print(f"\nResultados de train guardados en: {file_name}", end='')

    file_name = save_results_to_csv(main_image_paths=main_image_paths, ref_image_paths=ref_image_paths, questions=questions, refs=test_refs, hyps=test_hyps, epoch=epoch, exp_dirpath=EXP_DIR_PATH, split="test")
    print(f"\nResultados de test guardados en: {file_name}\n")

    # Calculate metrics
    gts = {k: [{ 'caption': v }] for k, v in enumerate(test_refs)}
    res = {k: [{ 'caption': v }] for k, v in enumerate(test_hyps)}

    evaluator = Evaluator()
    evaluator.do_the_thing(gts, res)

    metrics_table = Evaluator.metrics_to_log(evaluator.evaluation_report, train_loss, test_loss, train_hnm_loss)

    if best_bleu1 < evaluator.evaluation_report['Bleu_1']:
        try:
            os.remove(EXP_DIR_PATH + "/best_bleu1_" + str(epoch_best_bleu1) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_bleu1 = evaluator.evaluation_report['Bleu_1']
        epoch_best_bleu1 = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_bleu1_" + str(epoch) + "_model.pt")

    if best_cider < evaluator.evaluation_report['CIDEr']:
        try:
            os.remove(EXP_DIR_PATH + "/best_cider_" + str(epoch_best_cider) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_cider = evaluator.evaluation_report['CIDEr']
        epoch_best_cider = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_cider_" + str(epoch) + "_model.pt")

    if best_meteor < evaluator.evaluation_report['METEOR']:
        try:
            os.remove(EXP_DIR_PATH + "/best_meteor_" + str(epoch_best_meteor) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_meteor = evaluator.evaluation_report['METEOR']
        epoch_best_meteor = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_meteor_" + str(epoch) + "_model.pt")

    if best_rougel < evaluator.evaluation_report['ROUGE_L']:
        try:
            os.remove(EXP_DIR_PATH + "/best_rougel_" + str(epoch_best_rougel) + "_model.pt")
        except OSError as e:
            print("No se ha eliminado nada")
        best_rougel = evaluator.evaluation_report['ROUGE_L']
        epoch_best_rougel = epoch
        torch.save(model.state_dict(), EXP_DIR_PATH + "/best_rougel_" + str(epoch) + "_model.pt")

    # if best_bertscore < test_metrics['BertScore']:
    #     try:
    #         os.remove(EXP_DIR_PATH + "/best_bertscore_" + str(epoch_best_bertscore) + "_model.pt")
    #     except OSError as e:
    #         print("No se ha eliminado nada")
    #     best_bertscore = test_metrics['BertScore']
    #     epoch_best_bertscore = epoch
    #     torch.save(model.state_dict(), EXP_DIR_PATH + "/best_bertscore_" + str(epoch) + "_model.pt")

    # if best_f1_chexbert < test_metrics['F1 ChexBert']:
    #     try:
    #         os.remove(EXP_DIR_PATH + "/best_f1_chexbert_" + str(epoch_best_f1_chexbert) + "_model.pt")
    #     except OSError as e:
    #         print("No se ha eliminado nada")
    #     best_f1_chexbert = test_metrics['F1 ChexBert']
    #     epoch_best_f1_chexbert = epoch
    #     torch.save(model.state_dict(), EXP_DIR_PATH + "/best_f1_chexbert_" + str(epoch) + "_model.pt")

    lr_scheduler.step(test_loss)

    # Unfreeze encoder wights after 2 epochs
    # if epoch == 2:
    #     for param in model.encoder.parameters():
    #         param.requires_grad = True

    EXP_DIR_PATH
    # Open the file in write mode ('w')
    with open(EXP_DIR_PATH + "/log.txt", 'a') as file:
        # Write the string to the file
        file.write("EPOCH: " + str(epoch) + "\n")
        file.write(metrics_table + "\n\n")
    print(f'Metrics saved in {EXP_DIR_PATH}/log.txt\n')

# Save Final weights
torch.save(model.state_dict(), EXP_DIR_PATH + "/last_model.pt")
