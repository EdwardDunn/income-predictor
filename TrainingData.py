'''
Created on 6 Jun 2018

@author: Edward Dunn
'''

import pandas as pd

SHUFFLED_DATA_PATH = "dataset-shuffled.csv"
DATA_PATH = "dataset.csv"

def shuffle_dataset():
    print("Shuffling dataset")
    data_path = "dataset.csv"         
    shuffled_data_path = "dataset-shuffled.csv"

    # Shuffle dataset
    fid = open(DATA_PATH, "r")
    li = fid.readlines()
    fid.close()  
    random.shuffle(li)
    
    # Create shuffled dataset
    fid = open(SHUFFLED_DATA_PATH, "w")
    fid.writelines(li)
    fid.close()
