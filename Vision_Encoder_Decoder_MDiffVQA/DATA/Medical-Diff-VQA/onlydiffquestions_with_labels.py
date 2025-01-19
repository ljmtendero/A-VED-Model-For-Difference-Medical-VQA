import pandas as pd
import numpy as np
from tqdm import tqdm


# NEW_COD_LABELS = {
#     -1.0: 1,
#     0.0: 2,
#     1.0: 3
# }

NEW_COD_LABELS = {
    -1.0: 1,
    0.0: 0,
    1.0: 1
}


def get_subject_and_study_ids(path_to_image):
    """
    Given a path to image, return the patient and subject ids.
    """
    _, patient_id, subject_id, _ = path_to_image.split('/')

    return int(patient_id[1:]), int(subject_id[1:])


def codificate_labels(labels):
    """
    Given a list of labels, return the codified labels.
    """

    return [0 if np.isnan(label) else NEW_COD_LABELS[label] for label in labels]


# Path to mimic_all.csv
mimic_all = 'mimic_all.csv'
df_all = pd.read_csv(mimic_all)

splits = ['train', 'val', 'test']
for split in splits:
    # Path to files
    mimic_pair_questions = f'medical_vqa_pair_onlydiffquestions_{split}.csv'

    # Load the data
    df_pair = pd.read_csv(mimic_pair_questions)

    new_df_pair = pd.DataFrame(columns=df_pair.columns.tolist() + ['main_image_labels', 'ref_image_labels'])

    break_for_loop = False
    for i, row in tqdm(iter(df_pair.iterrows()), total=len(df_pair), desc=split, unit='row'):
        path_to_images = row['images']
        main_img, ref_img = path_to_images.split(',')

        main_subject_id, main_study_id = get_subject_and_study_ids(main_img)
        ref_subject_id, ref_study_id = get_subject_and_study_ids(ref_img)

        main_image_labels = df_all[(df_all['subject_id'] == main_subject_id) & (df_all['study_id'] == main_study_id)].iloc[0].to_list()[2:-5]
        ref_image_labels = df_all[(df_all['subject_id'] == ref_subject_id) & (df_all['study_id'] == ref_study_id)].iloc[0].to_list()[2:-5]

        main_image_labels = codificate_labels(main_image_labels)
        ref_image_labels = codificate_labels(ref_image_labels)

        row['main_image_labels'] = main_image_labels
        row['ref_image_labels'] = ref_image_labels

        new_df_pair = new_df_pair._append(row)

        # for j, (main, ref) in enumerate(zip(main_image_labels, ref_image_labels)):
        #     if main == -1.0 and np.isnan(ref):
        #         print('Row:\t', i)
        #         columns = df_all.columns.tolist()[2:-5]
        #         print(columns)
        #         print('Main:\t', main_image_labels)
        #         print('Ref:\t', ref_image_labels)
        #         print('Anomaly:\t', columns[j], j)
        #         print('Report:\t', row['answer'])
        #         break_for_loop = True
        #         break

        # if break_for_loop:
        #     input('Press any key to continue...')
        #     print('\n\n\n')
        #     break_for_loop = False

    new_df_pair.to_csv(f'medical_vqa_pair_onlydiffquestions_{split}_with_labels2.csv', index=False)
    