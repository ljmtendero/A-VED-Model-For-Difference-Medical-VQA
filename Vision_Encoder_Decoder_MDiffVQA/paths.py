IMAGES_MIMIC_PATH = "/home/Data/datasets/mimic-cxr-jpg-512"

DICT_CSV_MIMIC_PATH = {
    # "train": "/home/maasala/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_questions_train_peque.csv",
    # "validation": "/home/maasala/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_questions_val_peque.csv",
    # "test": "/home/maasala/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_questions_test_peque.csv"

    # "train": "/home/maasala/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_questions_train.csv",
    # "validation": "/home/maasala/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_questions_val.csv",
    # "test": "/home/maasala/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_questions_test.csv"

    "train": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_train.csv",
    "validation": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_val.csv",
    "test": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_test.csv",

    # "train": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_train_with_labels2.csv",
    # "validation": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_val_with_labels2.csv",
    # "test": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_test_with_labels2.csv"
}

VOCAB_PATH = "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/vocab_diff.tgt"

SWINB_IMAGENET22K_WEIGHTS = "microsoft/swin-base-patch4-window12-384-in22k"
SWINB_IMAGENET22K_WEIGHTS_FINETUNE = "/home/ljmarten/RadiologyQA/swin_mimic/"

PATH_IDS_NO_RG_TRAIN = "/home/Data/datasets/mimic-cxr-jpg-512/TRAIN_SAMPLES_NO_rg.txt"
PATH_IDS_NO_RG_TEST = "/home/Data/datasets/mimic-cxr-jpg-512/TEST_SAMPLES_NO_rg.txt"

PATH_OPENI_REPORTS = "/home/Data/NEW/mimic-cxr/openi/ecgen-radiology/"       #No se usa
PATH_OPENI_IMGS = '/home/Data/NEW/mimic-cxr/openi/imgs/'       #No se usa

CHEXPERT_TRAIN_CSV = "/home/dparres/mychexpert/train.csv"       #No se usa
CHEXPERT_VALID_CSV = "/home/dparres/mychexpert/valid.csv"       #No se usa
CHEXPERT_IMAGES = "/home/dparres/mychexpert/"       #No se usa


DICT_MIMIC_OBS_TO_INT = {       #No se usa
    "enlarged cardiomediastinum": 0,      
    "cardiomegaly": 1,  
    "lung opacity": 2,       
    "lung lesion": 3,     
    "edema": 4,           
    "consolidation": 5,          
    "pneumonia": 6,          
    "atelectasis": 7,        
    "pneumothorax": 8,      
    "pleural effusion": 9,       
    "pleural other": 10,   
    "fracture": 11,      
    "support devices": 12,          
    "no finding": 13
}

DICT_MIMIC_INT_TO_OBS = {       #No se usa
    0: "enlarged cardiomediastinum",      
    1: "cardiomegaly",  
    2: "lung opacity",       
    3: "lung lesion",     
    4: "edema",           
    5: "consolidation",          
    6: "pneumonia",          
    7: "atelectasis",        
    8: "pneumothorax",      
    9: "pleural effusion",       
    10: "pleural other",   
    11: "fracture",      
    12: "support devices",          
    13: "no finding"
}

DICT_CHEXPERT_INT_TO_OBS = {       #No se usa
    0: "no finding",
    1: "enlarged cardiomediastinum",
    2: "cardiomegaly",
    3: "lung opacity",
    4: "lung lesion",
    5: "edema",
    6: "consolidation",
    7: "pneumonia",
    8: "atelectasis",
    9: "pneumothorax",
    10: "pleural effusion",
    11: "pleural other",
    12: "fracture",
    13: "support devices"
}