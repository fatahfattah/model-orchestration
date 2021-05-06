import sys
sys.path.append('..')

import torch
import torchvision.transforms as transforms

import numpy as np
from sklearn.cluster import DBSCAN
from dataloader import load_input

class PC_net():
    """
    Pixel clustering classifier
    Predict the number of pixel clusters that an image contains
    Clustering algorithm used is DBSCAN
    """

    def __init__(self):
        self.name = "Pixel clustering classifier"
        self.small_name = "pc"
        self.description = "Predict the number of pixel clusters that an image contains"
        self.input_type = "image"
        self.inference_type = "clustering"

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
        output = "none"

        if n_clusters == 1:
            output = "one"
        elif n_clusters > 1:
            output = "multi"

        print(f"{self.small_name} output: {output}@{n_clusters}")
        return output


if __name__ == "__main__":
    pc = PC_net()
    image = load_input("image", "../example_image.tif")
    
    print(pc.infer(image))