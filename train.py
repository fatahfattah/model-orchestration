from torchvision import transforms
from models.not_drawing import NOTDRAWING_NN
from models.drawing import DRAWING_NN
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from collections import Counter
from shutil import copy
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)
from torch.utils.data import DataLoader, Subset


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


def specialized_train(trainset, testset, path, modelname, train_transforms, test_transforms):
    """
    Train a neural network on a specialized dataset, saves it to modelname at given path
    """

    #Instantiate torch data loaders
    train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Choose cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Init model architecture and move to cpu/gpu
    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False, force_reload=True)
    model.fc = nn.Linear(2048, len(classes))
    model.to(device)

    # Define losses
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    # Define number of epochs
    epochs = 10

    # Starting training
    training_losses = []
    validation_losses = {}
    print(f"Starting model training for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        model.train()
        print(f"Epoch: {epoch+1}...")
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = train_transforms(inputs)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
        
        training_losses.append(running_loss/len(train_data_loader))

        outs = []
        true_labels = []
        model.eval()
        # Validate model over unseen data and print accuracy for each class
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for data in test_data_loader:
                images, labels = data[0].to(device), data[1].to(device)
                images = test_transforms(images)
                true_labels.extend(labels)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                outs.extend(predicted)
                c = (predicted == labels).squeeze()
                for i, label in enumerate(labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(len(classes)):
            try:
                loss = classes[i], 100 * class_correct[i] / class_total[i]
                validation_losses.setdefault(loss[0], []).append(loss[1])
                print('Validation accuracy of %5s : %2d %%' % (loss))
            except ZeroDivisionError as e:
                print(f"Division by zero, not enough validation data.")

        true_labels = [int(x) for x in true_labels]
        outs = [int(x) for x in outs]
        print(classification_report(true_labels, outs))
    
    # # Save trained model to storage
    PATH = f'{path}/{modelname}.pth'
    torch.save(model.state_dict(), PATH)

    return (training_losses, validation_losses)

if __name__ == "__main__":
    input_type = 'image'

    target_classifier = CNCMANY_NN()
    drawing = DRAWING_NN()
    not_drawing = NOTDRAWING_NN()

    image_size = (299, 299)
    batch_size = 8
    data_dir = "/home/fatah/Desktop/cnc_subset"
    dest_dir = "/home/fatah/Desktop/filtered_validation_relative_rank"
    model_identifier = "cncmany_drawingfilter_relative_rank"
    model_save_path = "./models/"
    with_filter = True

    # Load dataset
    dataset = ImageFolderWithPaths(data_dir, transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Resize(image_size)]))


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

    filters = {
		'manychemical': [drawing, 'drawing'],
		'onechemical': [not_drawing, 'not_not_drawing'],
		'nonchemical': [not_drawing, 'not_drawing']
    }

    # Filter out images according to defined filters from classifier relations
    indices = np.arange(len(trainset))

    # We try to filter out indices that are not in our filters
    train_filtered_indices = []
    train_negative_filtered_indices = []
    for i in tqdm(indices):
        # Retrieve the class label from the trainset
        truth_label =  classes[trainset[i][1]]

        # If the output from the filter classifier is equal to the target filter label, then we include this indice in the trainset
        if  filters[truth_label][0].infer(load_input(input_type, trainset[i][2])) == filters[truth_label][1]:
            train_filtered_indices.append(i)
        else:
            train_negative_filtered_indices.append(i)

    positive_trainset = Subset(trainset, train_filtered_indices)
    negative_trainset = Subset(trainset, train_negative_filtered_indices)

    # Filter out images according to defined filters from classifier relations
    indices = np.arange(len(testset))
    
    # # We try to filter out indices that are not in our filters
    test_filtered_indices = []
    test_negative_filtered_indices = []
    for i in tqdm(indices):
        # Retrieve the class label from the testset
        truth_label =  classes[testset[i][1]]
        input_path = testset[i][2]
        # If the output from the filter classifier is equal to the target filter label, then we include this indice in the trainset
        if  filters[truth_label][0].infer(load_input(input_type, input_path)) == filters[truth_label][1]:
            test_filtered_indices.append(i)
        else:
            test_negative_filtered_indices.append(i)

    positive_testset = Subset(testset, test_filtered_indices)
    negative_testset = Subset(testset, test_negative_filtered_indices)
    

    print(f"Number of training datapoints after filter: {len(positive_trainset)}")
    print(Counter([classes[dataset.targets[i]] for i in train_filtered_indices]))

    print(f"Number of negative training datapoints after filter: {len(negative_trainset)}")
    print(Counter([classes[dataset.targets[i]] for i in train_negative_filtered_indices]))

    print(f"Number of test datapoints after filter: {len(positive_testset)}")
    print(Counter([classes[dataset.targets[positive_testset.dataset.indices[i]]] for i in test_filtered_indices]))

    print(f"Number of negative test datapoints after filter: {len(negative_testset)}")
    print(Counter([classes[dataset.targets[negative_testset.dataset.indices[i]]] for i in test_negative_filtered_indices]))

    # We train on the filtered data
    training_losses, validation_losses = specialized_train(positive_trainset
                                                         ,positive_testset
                                                         ,model_save_path
                                                         ,model_identifier
                                                         ,target_classifier.train_preprocessing
                                                         ,target_classifier.preprocessing) 
    plt.figure()
    plt.plot(training_losses)

    plt.figure()
    for name, loss in validation_losses.items():
        plt.plot(loss)

    # We train the negative version of the specialized classifier
    training_losses, validation_losses = specialized_train(negative_trainset
                                                        , negative_testset
                                                        , model_save_path
                                                        , f"negative_{model_identifier}"
                                                        ,target_classifier.train_preprocessing
                                                        ,target_classifier.preprocessing)
    plt.figure()
    plt.plot(training_losses)

    plt.figure()
    for name, loss in validation_losses.items():
        plt.plot(loss)

    plt.show()