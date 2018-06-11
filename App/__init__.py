'''
Created on 11 Jun 2018

@author: Edward Dunn

Description:
Module used to run the different functions of the application. A simple console user interface is
used to allow the user choose which function to run.

'''

import tensorflow as tf
import os
import Tkinter, tkFileDialog

import App.Estimator


def create_model():
    # Set tensor flow logging level
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    #run_estimator()
    train_model_with_cross_validation()
    
def test_adult(filePath):
    # Set tensor flow logging level
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    analyse_test_adult(filePath)
    
# Simple console user interface
print('1. Create, train and test model')
print('2. Test model with single vector\n')
userInput = input('Please select an option\n')

if userInput == 1:
    create_model()
elif userInput ==2:
    # Close the root window
    Tkinter.Tk().withdraw()
    # Open file dialog box to choose test image
    file_path = tkFileDialog.askopenfilename()
    test_adult(file_path)
else:
    print('Please specify an option')