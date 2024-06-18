# AI-assigment-dog-species

## Run with pyenv

## Data preparation

### Steps to prepare dataset

- Make sure to install the pip dependencies

    ```
    cd data_preparaion
    pip install -r requirements
    ```

- Scrap google image for data

    `python scrapper.py`

    Scraps a dataset of 10 different dog breeds

- Standadize dataset to images of 256x256

    `python standardize.py`

- Split dataset into training, validation and testing sets

    `python split_data.py`

## Data modelling

### Steps to run data modeling

- Make sure to install the pip dependencies:

    ```
    cd ../data_modelling
    pip install -r requirements
    ```

- For a faster training process check if youre gpu is detected:

    `python gpu_detect.py`

- Run the training process:

    `python train.py`

    Trains for 50 epochs each of the 3 models (ResNet50, MobileNetV3, DenseNet121)

    The loss and accuracy is displayed at each epoch

    Between each model training you will have data visualization of the model you just trained

    The training time will be displayed at the end

## Data visualization

The loss/accuracy curves and confusion matrix of the previously trained models