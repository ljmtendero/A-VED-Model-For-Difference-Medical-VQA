# Read a csv and extract only the questions with question_type == 'difference'

import pandas as pd

splits = ['train', 'val', 'test']

for split in splits:
    # Read the csv
    df = pd.read_csv(f'medical_vqa_pair_questions_{split}.csv')

    # Extract only the questions with question_type == 'difference'
    df_diff = df[df['question_type'] == 'difference']

    # Save the extracted questions to a new csv
    df_diff.to_csv(f'medical_vqa_pair_onlydiffquestions_{split}.csv', index=False)