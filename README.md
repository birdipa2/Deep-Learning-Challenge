# Deep-Learning-Challenge

# Alphabet Soup Charity Deep Learning Model

This repository contains a deep learning model developed using a neural network to predict the success of funding applicants for Alphabet Soup, a charity organization. The model is trained on a dataset that includes various features related to each applicant, and the goal is to accurately classify whether an applicant will be successful or not.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

The deep learning model in this project aims to provide a classification solution for Alphabet Soup to optimize the allocation of resources. By training on historical data of funding applicants and their outcomes, the model learns patterns and makes predictions on new, unseen data.

## Data Preprocessing

In the data preprocessing step, the dataset is prepared for model training by performing the following tasks:
- Removing non-beneficial ID columns, such as EIN and NAME.
- Binning and replacing low-frequency values in the APPLICATION_TYPE and CLASSIFICATION columns to reduce noise in the data.
- Encoding categorical variables using one-hot encoding with `pd.get_dummies()`.
- Splitting the dataset into features (X) and target (y) arrays.
- Scaling the data using StandardScaler to normalize the numerical features.

## Model Architecture

The deep learning model architecture consists of a sequential neural network with the following layers:
1. Input layer: The number of neurons is determined by the number of input features.
2. Hidden layers: Two hidden layers with 80 and 32 neurons, respectively, and 'relu' activation function.
3. Output layer: One neuron with 'sigmoid' activation function to predict the binary outcome.

## Model Training and Evaluation

The model is compiled using the binary cross-entropy loss function, Adam optimizer, and accuracy metric. The training data is used to fit the model for a certain number of epochs, with the scaled features and target values. The model's performance is evaluated using the test data to calculate the loss and accuracy metrics.

## Model Optimization

To optimize the model performance, a separate code file, `AlphabetSoupCharity_Optimization.ipynb`, is provided. This code file includes the process of optimizing the model using the Keras Tuner. The Keras Tuner performs a hyperparameter search to find the best configuration for the neural network model. It searches over a defined search space of hyperparameters, such as the number of units in hidden layers and the learning rate for the optimizer, to improve the model's performance.

## Usage

To use this deep learning model, follow these steps:
1. Install the necessary dependencies mentioned in the project's requirements.txt file.
2. Download the dataset from [charity_data.csv](https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv) and place it in the same directory as the project files.
3. Run the preprocessing code (Deep_Learning_Challenge.ipynb) to clean and preprocess the dataset.
4. Run the model training and evaluation code (Deep_Learning_Challenge.ipynb) to train the model and assess its performance.
5. (Optional) Run the model optimization code (AlphabetSoupCharity_Optimization.ipynb) to optimize the model using the Keras Tuner.
6. Use the trained model to make predictions on new, unseen data.

## Code Files

- `Deep_Learning_Challenge.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `AlphabetSoupCharity_Optimization.ipynb`: Jupyter Notebook containing the code for optimizing the model using the Keras Tuner.

## Contributing

Contributions to this project are welcome. If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## Contact

Should you have any questions or concerns, please do not hesitate to contact me at param.birdi@utoronto.ca
