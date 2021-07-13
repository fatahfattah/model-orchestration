from models.not_drawing import NOTDRAWING_NN
from models.drawing import DRAWING_NN
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

import pickle
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)
from torch.utils.data import DataLoader, Subset

from shutil import copy
import os

from dataloader import load_input, ImageFolderWithPaths
from models.cncmany import CNCMANY_NN
from models.hl import HL_NN
from models.drawing import DRAWING_NN
from models.not_drawing import NOTDRAWING_NN

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

"""
Program to (Re-)train models in a network with a meta-level filter using explicit rules based on the output of the models in our network
"""

if __name__ == "__main__":
    input_type = 'image'

    target_classifier = CNCMANY_NN()
    drawing = DRAWING_NN()
    not_drawing = NOTDRAWING_NN()

    filters = {
                'manychemical': [not_drawing, 'not_not_drawing'],
               'onechemical': [not_drawing, 'not_not_drawing'],
               'nonchemical': [drawing, 'not_drawing']
               }

    image_size = (299, 299)
    batch_size = 8
    data_dir = '/home/fatah/Desktop/cnc_subset'
    dest_dir = "/home/fatah/Desktop/filtered_validation"

    # Load dataset
    dataset = ImageFolderWithPaths(data_dir,
                                                transform=target_classifier.train_preprocessing)

    classes = dataset.classes

    indices = np.arange(len(dataset))
    # Train/test split
    train_indices, test_indices = train_test_split(indices, 
                                            train_size = int(len(dataset)//len(dataset.targets)*0.8*len(dataset.targets)), 
                                            test_size = int(len(dataset)//len(dataset.targets)*0.2*len(dataset.targets)), 
                                            stratify = dataset.targets)
    
    # Make train/test subsets
    trainset = Subset(dataset, train_indices)
    testset = Subset(dataset, test_indices)

    # Filter out images according to defined filters from classifier relations
    indices = np.arange(len(testset))
    
    # # We try to filter out indices that are not in our filters
    filtered_indices = []
    for i in tqdm(indices):
        # Retrieve the class label from the testset
        truth_label =  classes[testset[i][1]]
        input_path = testset[i][2]
        # If the output from the filter classifier is equal to the target filter label, then we include this indice in the trainset
        if  filters[truth_label][0].infer(load_input(input_type, input_path)) == filters[truth_label][1]:
            filtered_indices.append(i)

            copy(input_path, os.path.join(dest_dir, truth_label))


    testset = Subset(testset, filtered_indices)

    print(f"Number of validation datapoints after filter: {len(testset)}")
    print(Counter([classes[dataset.targets[i]] for i in filtered_indices]))
