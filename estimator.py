'''
Created on 6 Jun 2018

Description:
Module used to create a deep neural network using TensorFlow. Functions for cross-validation and
evaluation included.

@author: Edward Dunn

@version: 1.0.0.0
'''

import tensorflow as tf
import os
import cv2


# Number of samples that are to be propagated through the network
BATCH_SIZE = 200

# Used to define the number of iterations to take before stopping the training
TRAIN_STEPS = 1000

# Used in K-Fold cross validation
K_FOLD_STEPS = 5

# Directory used to save model
MODEL_DIR = "./dnn-classifier1"

def train_model_with_cross_validation():   
    # Shuffle data set to ensure order is random
    shuffle_dataset()
    
    # Used to show average accuracy from all training
    finalAccuracy = 0.0
    
    # Used to hold the accuracy ratings from each of the cross validation training steps
    training_results = []   
    
    # Get total number of rows (image vectors) in data set
    numberDataSetRows = get_dataset_num_rows()
    
    # K-Fold cross validation parameters
    foldSize =  numberDataSetRows / K_FOLD_STEPS
    foldStart = 0
    foldEnd = foldSize
    
    # For each cross validation step create a list of the testing rows to be tested
    for i in range(K_FOLD_STEPS):     
        
        testing_rows = []
        
        # Create a list beginning at foldStart and ending at foldEnd
        for j in range(int(foldStart), int(foldEnd)):
            testing_rows.append(j)
         
        # Ensure foldStart is not the same as the last folds end value
        foldStart+= foldSize+1
        foldEnd+= foldSize
        
        print("Cross validation step: " , i + 1)
        
        # Train the model
        # Pass the list of rows to be used for testing
        evalResult = run_estimator(testing_rows)
        
        # Add accuracy result returned from training to the training_results list
        training_results.append(float('{accuracy:.3f}'.format(**evalResult)))
 
    # Find average of training accuracy results
    for accuracyValue in training_results:
        finalAccuracy+= accuracyValue
    
    finalAccuracy = finalAccuracy / K_FOLD_STEPS  
    
    # Display final accuracy result
    print("\nTesting accuracy: " ,finalAccuracy)
    
    
def run_estimator(testing_rows):
  
    global BATCH_SIZE
    global TRAIN_STEPS
     
    print("Training model")

    training_data_path = "trainingset.csv" 
    testing_data_path = "testingset.csv"
    
    # If training and testing data sets already exist, delete them
    if os.path.exists(training_data_path) and os.path.exists(testing_data_path):
        os.remove(training_data_path)
        os.remove(testing_data_path)
    
    # Fetch and parse data
    (trainFeatures, trainLabels), (testFeatures, testLabels) = TrainingData.load_data(testing_rows)
    
    # Get each feature from the FeatureLabels module and add it to the featureColumns array
    featureColumns = []  
    for feature in get_feature_labels():
        # Ensure the label name 'emotion' is not used
        if(feature != 'emotion'):
            featureColumns.append(tf.feature_column.numeric_column(key=feature))
    
    # Using a fully connected neural network (deep neural network)
    # The file path is specified so that classifier has persistence
    classifier = tf.estimator.DNNClassifier(
        feature_columns=featureColumns,
        # Three hidden layers 
        hidden_units=[60, 40, 20],
        # The model must choose between 7 classes.
        n_classes=7, model_dir=MODEL_DIR)
    
    # Call the estimators train function and pass training data through it
    # Stop training after the specified number of TRAIN_STEPS (this is 1000)
    classifier.train(
        input_fn=lambda:TrainingData.train_input_fn(trainFeatures, trainLabels,
                                                 BATCH_SIZE),
        steps=TRAIN_STEPS)
    
    # Call the estimators evaluate function to evaluate the model
    evalResult = classifier.evaluate(
        input_fn=lambda:TrainingData.eval_input_fn(testFeatures, testLabels,
                                                BATCH_SIZE))
    
    return evalResult