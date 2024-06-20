# AI-assigment-dog-species

Training, dataset and visualization about dog breeds identification.

| Labrador Retriever | Golden Retriever | German Shepherd | Beagle | French Bulldog | Greyhound | Bulldog | Poodle | Rottweiler | Yorkshire Terrier |
| :----------------: | :-------------: | :-------------: | :----: | :------------: | :-------: | :-----: | :----: | :--------: | :---------------: |
| [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Labrador%2BRetriever/0.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Golden%2BRetriever/10.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/German%2BShepherd/100.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Beagle/10.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/French%2BBulldog/10.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Greyhound/10.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Bulldog/102.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Poodle/106.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Rottweiler/105.jpg" width="60px;"/><br /><sub>]() | [<img src="https://github.com/titi0267/AI-assigment-dog-species/blob/main/data_preparation/split_dataset/train/Yorkshire%2BTerrier/115.jpg" width="60px;"/><br /><sub>]() |

# How to run the project

## Run with pyenv

Create your python env, see pyenv documentation for more informations.

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

# Authors

|                                               @titi0267                                               | @twillsonepitech | @Antweneee |
| :----------------------------------------------------------------------------------------------------: | :-------: | :----: |
| [<img src="https://github.com/titi0267.png" width="60px;"/><br /><sub>](https://github.com/titi0267) | [<img src="https://github.com/twillsonepitech.png" width="60px;"/><br /><sub>](https://github.com/twillsonepitech) | [<img src="https://github.com/Antweneee.png" width="60px;"/><br /><sub>](https://github.com/Antweneee) |
