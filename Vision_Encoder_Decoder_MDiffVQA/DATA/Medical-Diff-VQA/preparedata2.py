import pandas as pd
import os
import glob

# Load the CSV files
pair_questions_path = "mimic_pair_questions.csv"
all_metadata_path = "mimic_all.csv"

pair_data = pd.read_csv(pair_questions_path)
all_metadata = pd.read_csv(all_metadata_path)

# Step 1: Create a new column "images" with the required path logic
def generate_image_paths(row, index, total_rows):
    base_path = "/home/Data/NEW/mimic-cxr/2.0.0/files_jpg_512/files/"
    subject_id = str(row["subject_id"])
    study_id = str(row["study_id"])
    ref_id = str(row.get("ref_id", ""))
    
    # Helper to retrieve the correct image from mimic_all.csv
    def get_correct_image(subject_id, study_id):
        matching_row = all_metadata[
            (all_metadata["subject_id"] == int(subject_id)) &
            (all_metadata["study_id"] == int(study_id))
        ]
        if not matching_row.empty:
            return matching_row.iloc[0]["dicom_id"]
        return None

    if row["question_type"] == "difference":
        # Get images for both study_id and ref_id
        study_image = get_correct_image(subject_id, study_id)
        ref_image = get_correct_image(subject_id, ref_id)
        
        study_path = os.path.relpath(
            os.path.join(base_path, f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{study_image}.jpg"), base_path
        ) if study_image else None
        
        ref_path = os.path.relpath(
            os.path.join(base_path, f"p{subject_id[:2]}/p{subject_id}/s{ref_id}/{ref_image}.jpg"), base_path
        ) if ref_image else None
        
        # Return relative paths if they exist
        return ",".join(filter(None, [study_path if study_path and os.path.exists(os.path.join(base_path, study_path)) else None,
                                      ref_path if ref_path and os.path.exists(os.path.join(base_path, ref_path)) else None]))
    else:
        # Get image for study_id only
        correct_image = get_correct_image(subject_id, study_id)
        if correct_image:
            image_path = os.path.relpath(
                os.path.join(base_path, f"p{subject_id[:2]}/p{subject_id}/s{study_id}/{correct_image}.jpg"), base_path
            )
            if os.path.exists(os.path.join(base_path, image_path)):
                return image_path
        return ""

# Track the progress of processing
total_rows = len(pair_data)
for idx, row in pair_data.iterrows():
    pair_data.at[idx, "images"] = generate_image_paths(row, idx, total_rows)
    if idx % 100 == 0 or idx == total_rows - 1:
        print(f"Processed {idx + 1}/{total_rows} questions")

# Step 2: Reorder columns and save to new CSV
columns_order = ["question_type", "question", "answer", "images", "split"]
pair_data = pair_data[columns_order]
pair_data.to_csv("medical_vqa_pair_questions.csv", index=False)

# Step 3: Split and save into separate files based on "split"
for split_value in ["train", "val", "test"]:
    split_data = pair_data[pair_data["split"] == split_value]
    split_data = split_data[["question_type", "question", "answer", "images"]]
    split_data.to_csv(f"medical_vqa_pair_questions_{split_value}.csv", index=False)
