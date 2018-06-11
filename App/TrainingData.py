'''
Created on 6 Jun 2018

@author: Edward Dunn
'''

import pandas as pd

import App.FeatureLabels


CSV_COLUMN_NAMES = get_feature_labels()

DATA_PATH = "dataset.csv"

def get_dataset_num_rows():
    return sum(1 for line in open(DATA_PATH))

def get_data(testing_rows):
    """ Creates the training and testing dataset files from the original data set """
    print("Creating training & testing dataset files")
    
    # data_path = "FeatureDetection/dataset.csv"         
    train_path = "trainingset.csv"
    test_path = "testingset.csv"
    
    # Create training data set
    dataFile = open(DATA_PATH)
    trainingFile = open(train_path, 'a')
    
    # Used to only write rows NOT in the testing_rows list
    trainingRowCounter = 0
    for line in dataFile.readlines():
        # Only add rows not specified in the testing K-Fold (testing_rows list)
        if trainingRowCounter not in testing_rows:
            trainingFile.write(line)
        
        trainingRowCounter += 1
    
    dataFile.close()
    trainingFile.close()
     
    # Create test data set
    dataFile = open(DATA_PATH)
    testingFile = open(test_path, 'a')
    
    # Used to only write rows IN the testing_rows list
    testRowCounter = 0
    for line in dataFile.readlines():
        # Only add rows specified in the testing K-Fold (testing_rows list)
        if testRowCounter in testing_rows:
            testingFile.write(line)
            
        testRowCounter += 1       
        
    dataFile.close()
    testingFile.close()
    
    return train_path, test_path

def load_data(testing_rows, labelName='emotion'):
    """Returns the dataset as (trainFeatures, trainLabels), (testFeatures, testLabels)."""
    print("Loading dataset")
    
    trainPath, testPath = get_data(testing_rows)

    # Parse training CSV file
    trainFile = pd.read_csv(trainPath, 
                        names=CSV_COLUMN_NAMES, # Column names
                         header=0) # Ignore the first row
    
    # Assign the DataFrames labels to trainLabels
    # Delete the labels from DataFrame
    # Assign remainder of the DataFrame to trainFeatures
    trainFeatures, trainLabels = trainFile, trainFile.pop(labelName)
    
    # Parse testing CVS file
    testFile = pd.read_csv(testPath, 
                       names=CSV_COLUMN_NAMES, # Column names e
                       header=0) # Ignore the first row
    
    # Assign the DataFrames labels to testLabels
    # Delete the labels from DataFrame
    # Assign remainder of the DataFrame to testFeatures
    testFeatures, testLabels = testFile, testFile.pop(labelName)

    return (trainFeatures, trainLabels), (testFeatures, testLabels)


def train_input_fn(features, labels, BATCH_SIZE):
    """An input function for training"""
    print("Training on input")
    
    # Convert the inputs to a Dataset
    # Converts the input features and labels into a tf.data.Dataset object
    # This is a high level TensorFlow API for reading data and transforming it into a form that the train method requires 
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle randomises the order
    # Buffer size must be more than the number of examples
    # Multiple batches are created using the batch size, increasing this reduces the training time
    dataset = dataset.shuffle(200).repeat().batch(BATCH_SIZE)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, BATCH_SIZE):
    """An input function for evaluation or prediction"""
    print("Evaluating input\n")
    
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert BATCH_SIZE is not None, "BATCH_SIZE must not be None"
    dataset = dataset.batch(BATCH_SIZE)

    # Return the dataset.
    return dataset