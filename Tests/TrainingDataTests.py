'''
Created on 21 Jun 2018

@author: Edward Dunn

Description:
Unit tests to ensure test methods used in the TrainingData class.

'''

import unittest

from App.TrainingData import get_dataset_num_rows

class TrainingDataTests(unittest.TestCase):
    
    def get_dataset_num_rows_ValidFile_NumberOfRowsReturned(self):
        # arrange
        # this hard coded number is taken from the adults.csv dataset file used
        # must be updated if revised
        expectedResult = 32562
        actualResult = 0
        
        # act
        actualResult = get_dataset_num_rows()
        
        # assert   
        self.assertEqual(expectedResult, actualResult, "Returned dataset row count incorrect, check dataset file used")


if __name__ == '__main__':
    unittest.main()