'''
Created on 6 Jun 2018

@author: Edward Dunn
'''

import pandas as pd

DATA_PATH = "dataset.csv"

def get_dataset_num_rows():
    return sum(1 for line in open(DATA_PATH))

def get_data(testing_rows):
    """ Creates the training and testing dataset files from the original data set """
    print("Creating training & testing dataset files")
    
    # data_path = "FeatureDetection/dataset.csv"         
    train_path = "trainingset.csv"
    test_path = "Ftestingset.csv"
    
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