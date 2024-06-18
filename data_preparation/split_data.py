import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define constants
DATASET_DIR = './standardized_dataset'  # Path to your standardized dataset
OUTPUT_DIR = './split_dataset'  # Path to save the split dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_dirs(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, val_dir, test_dir

def split_dataset(input_dir, output_dir):
    train_dir, val_dir, test_dir = create_dirs(output_dir)
    all_files = []
    
    # Collect all image files
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            all_files.append(os.path.join(subdir, file))
    
    # Split files into train, val, and test
    train_files, temp_files = train_test_split(all_files, test_size=1-TRAIN_RATIO, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=TEST_RATIO/(TEST_RATIO + VAL_RATIO), random_state=42)
    
    def copy_files(file_list, destination_dir):
        for file_path in file_list:
            relative_path = os.path.relpath(file_path, input_dir)
            destination_path = os.path.join(destination_dir, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(file_path, destination_path)
    
    # Copy files to their respective directories
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

# Run the dataset splitting
split_dataset(DATASET_DIR, OUTPUT_DIR)
