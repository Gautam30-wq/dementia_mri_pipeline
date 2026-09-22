import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

def generate_subject_splits(data_dir: str, output_json_path: str, n_splits: int = 5) -> None:
    """
    Parses the raw image directory, extracts subject identifiers, and generates 
    leakage-free cross-validation splits.
    """
    records = []
    classes = ['No Impairment', 'Very Mild Impairment', 'Mild Impairment', 'Moderate Impairment']
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Assumes the standard Kaggle/OASIS filename format: "SubjectID_SliceID.jpg"
                # If your dataset uses a different convention, adjust the split logic below.
                subject_id = file.split('_')[0] 
                records.append({
                    'filepath': os.path.join(class_dir, file),
                    'label': label_idx,
                    'subject_id': subject_id
                })
                
    df = pd.DataFrame(records)
    
    if df.empty:
        raise ValueError(f"No images found in {data_dir}. Check the path and directory structure.")

    # GroupKFold ensures the grouping variable (subject_id) remains entirely isolated per fold
    gkf = GroupKFold(n_splits=n_splits)
    splits = {}
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df['subject_id'])):
        splits[f"fold_{fold}"] = {
            "train": df.iloc[train_idx]['filepath'].tolist(),
            "test": df.iloc[test_idx]['filepath'].tolist()
        }
        
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Successfully generated {n_splits}-fold subject-level splits at: {output_json_path}")


class DementiaMRIDataset(Dataset):
    """
    PyTorch Dataset for multi-stage dementia classification.
    """
    def __init__(self, split_filepaths: list, transform=None):
        self.filepaths = split_filepaths
        self.transform = transform
        
        # Mapping aligned with the sequential disease stages
        self.label_map = {
            'No Impairment': 0, 
            'Very Mild Impairment': 1, 
            'Mild Impairment': 2, 
            'Moderate Impairment': 3
        }

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> tuple:
        path = self.filepaths[idx]
        class_name = os.path.basename(os.path.dirname(path))
        label = self.label_map[class_name]
        
        # Ensure structural MRI is loaded as grayscale to standardise input channels
        image = Image.open(path).convert('L') 
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Execute this script locally to generate the partition JSON before moving to Colab
    generate_subject_splits(
        data_dir="data/raw", 
        output_json_path="data/splits/cv_splits.json", 
        n_splits=5
    )