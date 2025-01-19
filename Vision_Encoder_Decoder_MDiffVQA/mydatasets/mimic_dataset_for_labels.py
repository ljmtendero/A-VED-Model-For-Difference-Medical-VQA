import os
import torch
import json
import ast
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
import random
from einops import rearrange

from paths import IMAGES_MIMIC_PATH, DICT_CSV_MIMIC_PATH, PATH_IDS_NO_RG_TRAIN, PATH_IDS_NO_RG_TEST
from mydatasets.mydatasets_utils import ifcc_clean_report, vilmedic_collate

class mimic_Dataset(Dataset):

    def __init__(self, 
                 tokenizer = None,
                 partition = "train",
                 text_preprocessing="ifcc_clean_report",
                 multi_image=2):

        self.tokenizer = tokenizer
        self.partition = partition
        self.text_preprocessing = text_preprocessing if text_preprocessing is None else eval(text_preprocessing)
        self.multi_image = multi_image
        self.random_padding = self.partition == "train"

        # Load CSV partition
        self.csv_path = DICT_CSV_MIMIC_PATH[self.partition]
        self.dataset_df = pd.read_csv(self.csv_path)

        # Remove empty question or answer from self.dataset_df
        self.remove_empty_text()  
        
        # Set images path
        self.img_root_dir = pathlib.Path(IMAGES_MIMIC_PATH) if IMAGES_MIMIC_PATH is not None else pathlib.Path.cwd()

        # Get token ids
        self.bos_token = self.tokenizer("[BOS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.eos_token = self.tokenizer("[EOS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.pad_token = self.tokenizer("[PAD]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.boq_token = self.tokenizer("[BOQ]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.bml_token = self.tokenizer("[MAINLABELS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.brl_token = self.tokenizer("[REFLABELS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    def __len__(self):
        return len(self.dataset_df)
    
    def clean_bad_ids_rg(self):
        print("Initial number of rows: ", self.dataset_df.shape[0])
        self.l_no_ids_rg = []
        with open(self.path_no_ids_rg, 'r') as file:
            for line in file:
                # Process each line as needed
                self.l_no_ids_rg.append(int(line.strip()))
                
        self.dataset_df.drop(self.l_no_ids_rg, inplace=True)
        print("Number of rows after deleting bad IDs: ", self.dataset_df.shape[0])


    def __getitem__(self, idx):
        # QA
        # Obtén la pregunta y la respuesta por separado
        question = self.dataset_df.iloc[idx].question
        question = self.text_preprocessing(question)
        question = self.tokenizer(
            question,
            padding=False,
            truncation=True,
            max_length=64,
            return_tensors="pt",
            add_special_tokens=False)['input_ids'][0]
        
        # Aplica el preprocesamiento al texto de salida (answer)
        raw_answer = self.dataset_df.iloc[idx].answer
        raw_answer = self.text_preprocessing(raw_answer)
        answer = self.tokenizer(
            raw_answer,
            padding=False,
            truncation=True,
            max_length=64,
            return_tensors="pt",
            add_special_tokens=False)['input_ids'][0]
        answer = torch.nn.functional.pad(answer, (0, 75 - len(answer) - 1), value=self.pad_token.item())
        answer = torch.cat([answer, self.eos_token])

        # Labels
        main_images_labels = ast.literal_eval(self.dataset_df.iloc[idx].main_image_labels)
        main_images_labels = self.labels_to_natural_language(main_images_labels)
        main_images_labels = self.text_preprocessing(main_images_labels)
        main_images_labels = self.tokenizer(
            main_images_labels,
            padding=False,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            add_special_tokens=False)['input_ids'][0]

        ref_images_labels = ast.literal_eval(self.dataset_df.iloc[idx].ref_image_labels)      
        ref_images_labels = self.labels_to_natural_language(ref_images_labels)
        ref_images_labels = self.text_preprocessing(ref_images_labels)
        ref_images_labels = self.tokenizer(
            ref_images_labels,
            padding=False,
            truncation=True,
            max_length=128,
            return_tensors="pt",
            add_special_tokens=False)['input_ids'][0]

        # Concatenate all the inputs
        input_ids = torch.cat([
            self.boq_token, question, 
            self.bml_token, main_images_labels, 
            self.brl_token, ref_images_labels])
        input_ids = torch.nn.functional.pad(input_ids, (0, 95 - len(input_ids)), value=self.pad_token.item())
        input_ids = torch.cat([input_ids, self.bos_token])

        question_ignore = torch.full((len(input_ids) - 1,), -100)
        labels = torch.cat([question_ignore, self.bos_token, answer])
        
        input_ids_with_answer = torch.cat([input_ids, answer])
        input_ids_with_answer_attention_mask = torch.ones_like(input_ids_with_answer)
        input_ids_with_answer_attention_mask[input_ids_with_answer == self.pad_token] = 0
                      
        return {
            'idx': idx,
            'input_ids': input_ids,
            'input_ids_with_answer': input_ids_with_answer,
            'input_ids_with_answer_attention_mask': input_ids_with_answer_attention_mask,
            'labels': labels,
            'answer': raw_answer,
        }
    
    def labels_to_natural_language(self, labels):
        """
        Given a list of labels, return the natural language labels.
        """
        
        anormalities = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged cardiomediastinum', 'fracture', 'lung lesion', 'lung opacity', 'no finding', 'pleural effusion', 'pleural other', 'pneumonia', 'pneumothorax', 'support devices']

        diagnosis = ""
        for i, label in enumerate(labels):
            if label == 1:
                diagnosis += f'finding {anormalities[i]}, '
            elif label == 0:
                diagnosis += f'missing {anormalities[i]}, '

        return diagnosis[:-2]
    
    def remove_empty_text(self):
        # Remove rows with empty question or answer
        self.dataset_df.dropna(subset=['question', 'answer'], inplace=True)
        print("Len before removing empty text", len(self.dataset_df))

        # Further remove any rows where the answer or question is effectively empty
        self.dataset_df = self.dataset_df[
            (self.dataset_df['question'].str.strip() != '') & 
            (self.dataset_df['answer'].str.strip() != '')
        ]
        print("Len after removing empty text", len(self.dataset_df))

    def get_collate_fn(self):
        def collate_fn(batch):
            idx = pytorch_default_collate([s['idx'] for s in batch])
            input_ids = pytorch_default_collate([s['input_ids'] for s in batch])
            input_ids_with_answers = pytorch_default_collate([s['input_ids_with_answer'] for s in batch])
            input_ids_with_answers_attention_masks = pytorch_default_collate([s['input_ids_with_answer_attention_mask'] for s in batch])
            labels = pytorch_default_collate([s['labels'] for s in batch])
            answers = [s['answer'] for s in batch]

            return {
                "idx": idx,
                "input_ids": input_ids,
                "input_ids_with_answers": input_ids_with_answers,
                "input_ids_with_answers_attention_masks": input_ids_with_answers_attention_masks,
                "labels": labels,
                "answers": answers,
            }
        return collate_fn
