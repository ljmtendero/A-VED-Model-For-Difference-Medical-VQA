import os
import torch
import json
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
                 transform, 
                 tokenizer,
                 processor,
                 partition = "train",
                 text_preprocessing="ifcc_clean_report",
                 multi_image=2):

        self.transform = transform
        self.tokenizer = tokenizer
        self.processor = processor
        self.partition = partition
        self.text_preprocessing = text_preprocessing if text_preprocessing is None else eval(text_preprocessing)
        self.multi_image = multi_image
        self.random_padding = self.partition == "train"
        # Set path for IDs without RG based on partition
        self.path_no_ids_rg = PATH_IDS_NO_RG_TRAIN if self.partition == "train" else PATH_IDS_NO_RG_TEST

        # Load CSV partition
        self.csv_path = DICT_CSV_MIMIC_PATH[self.partition]
        self.dataset_df = pd.read_csv(self.csv_path)

        print(f"The dataset contains {len(self.dataset_df)} entries.")
        # Remove entries where answer is "nothing has changed." if partition is "train" ONLY FOR RL
        # if self.partition == "train":
        #     self.dataset_df = self.dataset_df[self.dataset_df['answer'] != "nothing has changed."]

        print(f"The dataset contains {len(self.dataset_df)} entries.")
        
        # Remove empty question or answer from self.dataset_df
        self.remove_empty_text()

        # Set images path
        self.img_root_dir = pathlib.Path(IMAGES_MIMIC_PATH) if IMAGES_MIMIC_PATH is not None else pathlib.Path.cwd()

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

        #idx = 0
        img_list_from_idx = []

        num_images = len(self.dataset_df.iloc[idx].images.split(","))

        # Process all images from patient idx
        for i in range(num_images):
            img_name = self.img_root_dir / self.dataset_df.iloc[idx].images.split(",")[i]
            image = Image.open(img_name).convert('RGB')
            #image.save('rad.png')
            
            # Apply transformation
            if isinstance(self.transform, transforms.Compose):
                # If torchvision transformation
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                # If Albumentations transformation
                image = self.transform(image=np.asarray(image))['image']
            else:
                raise ValueError("Unknown transformation type. Supported types: torchvision.transforms.Compose, albumentations.core.composition.Compose")
            
            # Image Processor
            image = np.array(image)
            image = self.processor(image, random_padding=self.random_padding, size = 384, return_tensors="pt").pixel_values
            image = image.squeeze()

            # FOR PROCESSOR OF TORCHVISION
            # print("image shape: ", image.shape, 'type: ', type(image))
            # image = np.array(image)
            # image = torch.tensor(image, dtype=torch.float32)
            # image = rearrange(image, 'h w c -> c h w')
            # image = self.processor(image)

            #transforms.ToPILImage()(image[0]).save("trad.jpg")
            #print("max: ", torch.max(image))
            #print("min: ", torch.min(image))
            #print("------------------")

            # print(type(image))
                
            img_list_from_idx.append(image)

        # QA
            # Obtén la pregunta y la respuesta por separado
        question = self.dataset_df.iloc[idx].question
        question = self.text_preprocessing(question)
        
        # Aplica el preprocesamiento al texto de salida (answer)
        answer = self.dataset_df.iloc[idx].answer
        answer = self.text_preprocessing(answer)

        # Calculate images_mask
        im_and_immask = vilmedic_collate([img_list_from_idx], self.multi_image)
        images = im_and_immask["images"]
        images_mask = im_and_immask["images_mask"]
              
        return {'idx': idx, 'question': question, 'answer': answer, 'image': images, "images_mask": images_mask}
    
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

            images =  pytorch_default_collate([s['image'] for s in batch])
            images_mask = pytorch_default_collate([s['images_mask'] for s in batch])
            idx = pytorch_default_collate([s['idx'] for s in batch])
            questions = [s['question'] for s in batch]
            raw_answers = [s['answer'] for s in batch]


            # Tokenize questions
            questions = self.tokenizer(
                questions,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False
            )

            answers = self.tokenizer(
                raw_answers,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False
            )
                
            # # Construct the full question-answer strings
            # examples = [f"{question} [BOS] {answer} [EOS]" for question, answer in zip(questions, raw_answers)]
            
            # example = self.tokenizer(
            #     examples,
            #     max_length=128,
            #     padding='max_length',
            #     truncation=True,
            #     return_tensors='pt',
            #     add_special_tokens=False
            # )

            # # Tokenize questions
            # questions = self.tokenizer(
            #     questions,
            #     max_length=64,
            #     truncation=True,
            #     return_tensors='pt',
            #     add_special_tokens=False
            # )

            # answers = self.tokenizer(
            #     answers,
            #     max_length=64,
            #     truncation=True,
            #     return_tensors='pt',
            #     add_special_tokens=False
            # )

            # # Add token 0 at the beginning of the answer and token 3 at the end
            # answers.input_ids = torch.cat((torch.tensor([[0]]), answers.input_ids, torch.tensor([[3]])), dim=1)
            # answers.attention_mask = torch.cat((torch.tensor([[1]]), answers.attention_mask, torch.tensor([[1]]),), dim=1)
            # answers.type_ids = torch.cat((torch.tensor([[0]]), answers.type_ids, torch.tensor([[0]]),), dim=1)
            # # Now concat questions and answers
            # questions.input_ids = torch.cat((questions.input_ids, answers.input_ids), dim=1)
            # questions.attention_mask = torch.cat((questions.attention_mask, answers.attention_mask), dim=1)
            # questions.type_ids = torch.cat((questions.type_ids, answers.type_ids), dim=1)

            # #Now add padding to questions
            # max_len = 128
            # questions.input_ids = torch.cat((questions.input_ids, torch.zeros(questions.input_ids.shape[0], max_len - questions.input_ids.shape[1], dtype=torch.long)), dim=1)
            # questions.attention_mask = torch.cat((questions.attention_mask, torch.zeros(questions.attention_mask.shape[0], max_len - questions.attention_mask.shape[1], dtype=torch.long)), dim=1)
            # questions.type_ids = torch.cat((questions.type_ids, torch.zeros(questions.type_ids.shape[0], max_len - questions.type_ids.shape[1], dtype=torch.long)), dim=1)

            collated = {
                "idx": idx,
                "images": images,
                "questions_ids": questions.input_ids,
                "questions_mask": questions.attention_mask,
                "answers_ids": answers.input_ids,
                "answers_mask": answers.attention_mask,
                "images_mask": images_mask,
                "answers": raw_answers
            }
            
            return collated
        return collate_fn
