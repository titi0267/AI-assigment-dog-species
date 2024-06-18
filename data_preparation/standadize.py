import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (256, 256)
DATASET_DIR = './dataset'
OUTPUT_DIR = './standardized_dataset' 

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image

# Function to standardize dataset
def standardize_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subdir, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
            image = preprocess_image(file_path)
            relative_path = os.path.relpath(subdir, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            output_path = os.path.join(output_subdir, file)
            cv2.imwrite(output_path, (image * 255).astype(np.uint8))

# Run the standardization process
standardize_dataset(DATASET_DIR, OUTPUT_DIR)