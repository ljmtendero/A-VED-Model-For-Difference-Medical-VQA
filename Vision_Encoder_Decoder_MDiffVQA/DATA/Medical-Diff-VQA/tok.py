import sys
sys.path.append('/home/ljmarten/RadiologyQA/')

from paths import VOCAB_PATH
from transformers import BertTokenizer
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import ast


def ifcc_clean_report(report):
    cleaned_report = report.lower()
    cleaned_report = ' '.join(wordpunct_tokenize(report))
    return cleaned_report

def labels_to_natural_language(labels):
    """
    Given a list of labels, return the natural language labels.
    """
    
    anormalities = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    diagnosis = ""
    for i, label in enumerate(labels):
        if label == 1 or label == 3:
            diagnosis += f'finding {anormalities[i]}, '
        elif label == 2:
            diagnosis += f'missing {anormalities[i]}, '

    return diagnosis[:-2]


tokenizer = BertTokenizer(
    vocab_file=VOCAB_PATH,
    do_basic_tokenize=False, 
    use_fast=False, 
    max_length=128,
)
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[BOQ]', '[MAINLABELS]', '[REFLABELS]']
})

text_preprocessing = eval('ifcc_clean_report')

max_question_len = ('', float('-inf'))
max_answer_len = ('', float('-inf'))



splits = ['train', 'test']
for split in splits:
    df = pd.read_csv(f'medical_vqa_pair_onlydiffquestions_{split}_with_labels.csv')
    for i, row in tqdm(iter(df.iterrows()), desc=split, total=len(df), unit='rows'):
        # question = text_preprocessing(row['question'])
        # answer = text_preprocessing(row['answer'])

        # tokenized_question = tokenizer(
        #     question,
        #     # max_length=128,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt',
        #     add_special_tokens=False
        # )
        # tokenized_answer = tokenizer(
        #     answer,
        #     # max_length=128,
        #     padding='max_length',
        #     truncation=True,
        #     return_tensors='pt',
        #     add_special_tokens=False
        # )

        # actual_tokenized_question_len = len(tokenized_question['input_ids'][0])
        # actual_tokenized_answer_len = len(tokenized_answer['input_ids'][0])

        # if actual_tokenized_question_len > max_question_len[1]:
        #     max_question_len = (question, actual_tokenized_question_len)

        # if actual_tokenized_answer_len > max_answer_len[1]:
        #     max_answer_len = (answer, actual_tokenized_answer_len)

        main_image_labels = ast.literal_eval(row['main_image_labels'])
        ref_image_labels = ast.literal_eval(row['ref_image_labels'])

        main_image_labels = labels_to_natural_language(main_image_labels)
        ref_image_labels = labels_to_natural_language(ref_image_labels)

        main_image_labels = text_preprocessing(main_image_labels.lower())
        ref_image_labels = text_preprocessing(ref_image_labels.lower())

        tokenized_main_images_labels = tokenizer(
            main_image_labels,
            # max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        tokenized_ref_images_labels = tokenizer(
            ref_image_labels,
            # max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False
        )

        actual_tokenized_main_images_labels_len = len(tokenized_main_images_labels['input_ids'][0])
        actual_tokenized_ref_images_labels_len = len(tokenized_ref_images_labels['input_ids'][0])

        if actual_tokenized_main_images_labels_len > max_question_len[1]:
            max_question_len = (main_image_labels, actual_tokenized_main_images_labels_len)

        if actual_tokenized_ref_images_labels_len > max_answer_len[1]:
            max_answer_len = (ref_image_labels, actual_tokenized_ref_images_labels_len)

print(f'Max question len: {max_question_len}')
print(f'Max answer len: {max_answer_len}')

# print(f'Max question len: {max_question_len}')
# print(f'Max answer len: {max_answer_len}')