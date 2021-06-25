import sys
sys.path.append('..')

import torch
import torchvision.transforms as transforms

import numpy as np
from sklearn.cluster import DBSCAN
from dataloader import load_input

from classifier import Classifier

class PC_net(Classifier):
    """
    Pixel clustering classifier
    Predict the number of pixel clusters that an image contains
    Clustering algorithm used is DBSCAN
    """

    def __init__(self):
        super().__init__(["n_clusters"],
                         "Pixel clustering classifier",
                         "pc",
                         "Predict the number of pixel clusters that an image contains.",
                         "image",
                         "clustering")
        
    def infer(self, image):
        """
        Resizes the given image to 100x100, since DBSCAN on big images is too slow
        """
        image = np.array(image)
        image = np.resize(image, (100,100, 3))
        x, y, z = image.shape
        image = image.reshape(x * y, z)
        # Fit DBSCAN on the input, for which the number of labels correspond to the number of clusters
        db = DBSCAN(eps=0.3, min_samples=10).fit(image)
        labels = db.labels_
        # -1 label corresponds to noisy output, so we remove one count if the label is present
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Display
        print(f"{self.small_name}: {n_clusters} clusters")

        return str(n_clusters)