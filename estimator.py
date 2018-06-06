'''
Created on 6 Jun 2018

@author: Edward Dunn
'''

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