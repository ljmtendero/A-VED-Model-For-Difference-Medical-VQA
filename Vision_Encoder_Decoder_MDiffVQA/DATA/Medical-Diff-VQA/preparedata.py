import pandas as pd
import os
import glob

# Load the CSV file
file_path = "mimic_pair_questions.csv"
data = pd.read_csv(file_path)

# Step 1: Delete the column "ref_id" and rows where "question_type" is "difference"
data = data.drop(columns=["ref_id"])
data = data[data["question_type"] != "difference"]

# Step 2: Create a new column "images" with the required path logic
def generate_image_paths(row):
    base_path = "/home/Data/NEW/mimic-cxr/2.0.0/files_jpg_512/files/"
    subject_id = str(row["subject_id"])
    study_id = str(row["study_id"])
    path = f"p{subject_id[:2]}/p{subject_id}/s{study_id}/"
    
    # Get all image files in the directory
    full_path = os.path.join(base_path, path)
    if os.path.exists(full_path):
        image_files = glob.glob(os.path.join(full_path, "*.jpg"))
        image_files = [os.path.relpath(img, base_path) for img in image_files]
        return ",".join(image_files)
    return ""

data["images"] = data.apply(generate_image_paths, axis=1)

# Step 3: Reorder columns and save to new CSV
columns_order = ["question_type", "question", "answer", "images", "split"]
data = data[columns_order]
data.to_csv("medical_vqa_pair_questions.csv", index=False)

# Step 4: Split and save into separate files based on "split"
for split_value in ["train", "val", "test"]:
    split_data = data[data["split"] == split_value]
    split_data = split_data[["question_type", "question", "answer", "images"]]
    split_data.to_csv(f"medical_vqa_pair_questions_{split_value}.csv", index=False)
