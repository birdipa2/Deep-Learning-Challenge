# Overview of the Analysis:

The purpose of this analysis is to develop a deep learning model using a neural network to classify whether a funding applicant from Alphabet Soup will be successful. The model is trained on a dataset that contains various features related to each applicant, such as application type, classification, income amount, etc. The goal is to create a model that accurately predicts whether an applicant will be successful in order to optimize the allocation of resources by Alphabet Soup.

# Results:

## Data Preprocessing:

Target Variable: The target variable for the model is the "IS_SUCCESSFUL" column, which indicates whether an applicant was successful (1) or not (0).

Features: The features for the model include all the columns in the dataset except for the target variable. These features provide information about each applicant and are used to predict their success.

Variables to Remove: The "EIN" and "NAME" columns are removed from the input data as they do not provide any relevant information for the classification task.

## Compiling, Training, and Evaluating the Model:

Neurons, Layers, and Activation Functions: The initial neural network model consists of two hidden layers. The first hidden layer has 80 neurons with the 'relu' activation function, and the second hidden layer has 32 neurons with 'relu' activation. The output layer has 1 neuron with the 'sigmoid' activation function.

Target Model Performance: The target model performance is to accurately predict whether an applicant will be successful. The evaluation metric used is accuracy, which measures the proportion of correct predictions.

Steps to Increase Model Performance: To increase the model performance, the number of hidden layers were increased to 3 and number of epochs during training was increased from 100 to 200. Additionally, hyperparameter tuning using the Keras Tuner was performed to search for the optimal number of units in the hidden layers and the learning rate for the optimizer. This optimization process helped in finding the best configuration for the model.

Summary:

The initial deep learning model using a neural network consisted of two hidden layers with 80 and 32 neurons, respectively. This model achieved a certain level of success in predicting the success of funding applicants. However, the overall performance of the model could be improved.

The optimized model, which included three hidden layers with 100, 50, and 25 neurons, respectively, showed improved performance after hyperparameter tuning using the Keras Tuner. By increasing the number of epochs during training and finding the optimal configuration for the model, the accuracy and predictive power were enhanced.

In conclusion, while the initial model provided insights into the classification of funding applicants, the optimized model with three layers demonstrated improved accuracy. 





