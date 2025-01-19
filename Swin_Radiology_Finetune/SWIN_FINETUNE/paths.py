
IMAGES_MIMIC_PATH = "/home/DATA/mimic-cxr/2.0.0/files_jpg_512/files"

MIMIC_PATH_TRAIN = "/home/DATA/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/train.metadata.csv"
MIMIC_PATH_VAL = "/home/DATA/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/validate.metadata.csv"
MIMIC_PATH_TEST = "/home/DATA/mimic-cxr/2.0.0/RRG/mimic-cxr/findings/test.metadata.csv"

SWINB_IMAGENET22K_WEIGHTS = "microsoft/swin-base-patch4-window12-384-in22k"

DICT_MIMICALL_OBS_TO_INT = {      
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Enlarged Cardiomediastinum": 4,
    "Fracture": 5,
    "Lung Lesion": 6,
    "Lung Opacity": 7,
    "No Finding": 8,
    "Pleural Effusion": 9,
    "Pleural Other": 10,
    "Pneumonia": 11,
    "Pneumothorax": 12,
    "Support Devices": 13
}

DICT_MIMICALL_INT_TO_OBS = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Consolidation",
    3: "Edema",
    4: "Enlarged Cardiomediastinum",
    5: "Fracture",
    6: "Lung Lesion",
    7: "Lung Opacity",
    8: "No Finding",
    9: "Pleural Effusion",
    10: "Pleural Other",
    11: "Pneumonia",
    12: "Pneumothorax",
    13: "Support Devices"
}

# Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion, Pleural Other,Pneumonia,Pneumothorax,Support Devices

DICT_MIMIC_OBSKEY_TO_INT = {
    -2: 0,
    0: 0,
    1: 1,
    -1: 1
}