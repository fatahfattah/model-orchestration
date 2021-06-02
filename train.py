import argparse

import random
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataloader import load_input, ImageFolderWithPaths
from sklearn.model_selection import train_test_split
import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pickle
from tqdm import tqdm

from dataloader import load_input
from models.cnc import CNC_net
from models.hl import HL_net

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

"""
Program to (Re-)train models in a network with a meta-level filter using explicit rules based on the output of the models in our network
"""

if __name__ == "__main__":
    input_type = 'image'

    filters = {'chemical': ["programlisting", "drawing", "genesequence", "chemicalstructure"],
               'nonchemical': ["drawing", "graph"]}

    target_agent = CNC_net()
    aucillery_agent = HL_net()

    image_size = (299, 299)
    batch_size = 8
    data_dir = '/home/fatah/Desktop/thesis/repos/cic_playground/I20210114/dataset'

    # Load dataset
    dataset = ImageFolderWithPaths(data_dir,
                                                transform=target_agent.train_preprocessing)

    indices = np.arange(len(dataset))
    # Train/test split
    train_indices, test_indices = train_test_split(indices, train_size = 1000*2, test_size = 200*2, stratify = dataset.targets)
    

    # Make train/test subsets
    trainset = Subset(dataset, train_indices)
    testset = Subset(dataset, test_indices)
    # Filter out images according to defined filters from agent relations
    indices = np.arange(len(trainset))
    
    filtered_indices = [i for i in indices if aucillery_agent.infer(load_input(input_type, trainset[i][2])) in filters[target_agent.classes[trainset[i][1]]]]
    # pickle.dump(filtered_indices, open('filtered_indices.p', 'wb'))
    # filtered_indices = pickle.load(open('filtered_indices.p', 'rb'))
    trainset = Subset(trainset, filtered_indices)


    print(f"Number of training datapoints after filter: {len(trainset)}")

    #Instantiate torch data loaders
    train_data_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    classes = target_agent.classes

    # Choose cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Init model architecture and move to cpu/gpu
    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False, force_reload=True)
    model.fc = nn.Linear(2048, 2)
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
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            outputs, aux = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_losses.append(running_loss / 100)
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

        model.eval()
        # Validate model over unseen data and print accuracy for each class
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for data in test_data_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(len(classes)):
            loss = classes[i], 100 * class_correct[i] / class_total[i]
            validation_losses.setdefault(loss[0], []).append(loss[1])
            print('Validation accuracy of %5s : %2d %%' % (loss))

    # # Save trained model to storage
    PATH = './models/cnc_specialized.pth'
    torch.save(model.state_dict(), PATH)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(training_losses)

    plt.figure()
    for name, loss in validation_losses.items():
        plt.plot(loss)

    plt.show()