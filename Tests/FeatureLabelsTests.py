'''
Created on 6 Jun 2018

@author: Edward Dunn

Description:
Unit tests to ensure correct labels re returned from the FeatureLabels class.

'''

import unittest

from App.FeatureLabels import get_feature_labels

class TrainingDataTests(unittest.TestCase):
    
    def get_feature_labels_ValidArrayReturned(self):
        ## arrange
        expectedResult = ['<= 10.000', '<= 20.000', '<= 30.000', '<= 40.000', '<= 50.000', '> 50.000', 'label']
        actualResult = []
        
        ## act
        actualResult = get_feature_labels()
        
        ## assert   
        self.assertEqual(expectedResult, actualResult, "Returned feature label array does not match expected")


if __name__ == '__main__':
    unittest.main()