IMAGES_MIMIC_PATH = "/home/Data/datasets/mimic-cxr-jpg-512"

DICT_CSV_MIMIC_PATH = {

    "train": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_train.csv",
    "validation": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_val.csv",
    "test": "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/medical_vqa_pair_onlydiffquestions_test.csv",

}

VOCAB_PATH = "/home/ljmarten/RadiologyQA/DATA/Medical-Diff-VQA/vocab_diff.tgt"

SWINB_IMAGENET22K_WEIGHTS = "microsoft/swin-base-patch4-window12-384-in22k"
SWINB_IMAGENET22K_WEIGHTS_FINETUNE = "/home/ljmarten/RadiologyQA/swin_mimic/"

PATH_IDS_NO_RG_TRAIN = "/home/Data/datasets/mimic-cxr-jpg-512/TRAIN_SAMPLES_NO_rg.txt"
PATH_IDS_NO_RG_TEST = "/home/Data/datasets/mimic-cxr-jpg-512/TEST_SAMPLES_NO_rg.txt"