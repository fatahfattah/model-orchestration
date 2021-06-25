import sys
sys.path.append('..')
import numpy as np

from classifier import Classifier

class PC_net(Classifier):
    """
    Pixel clustering classifier
    Predict the number of pixel clusters that an image contains
    Clustering algorithm used is DBSCAN
    """

    def __init__(self):
        super().__init__(["zero", "one", "many"],
                         "Pixel clustering classifier",
                         "pc",
                         "Predict the number of pixel clusters that an image contains.",
                         "image",
                         "clustering")
        
    def infer(self, image, explore=False):
        image = image.resize((100, 100))
        image = image.convert("L")
        image = np.array(image)
        m = len(image)
        n = len(image[0])

        mat = []
        for i in range(m):
            mat.append(['o' if image[i][j] < 220 else 'x' for j in range(len(image[i]))])

        def expand(cluster_id, clusters, i, j, mat, k=3, pixels=None):
            if i > 0 and j > 0 and i < len(mat) and j < len(mat[0]):
                if mat[i][j] == 'o':
                    clusters[cluster_id].append((i, j))
                    mat[i][j] = '@'
                    for d in range(i-k, i+k):
                        for l in range(j-k, j+k):
                            expand(cluster_id, clusters, d, l, mat, k, pixels)
        
        try:
            clusters = {}
            pixels = []
            for i in range(m):
                for j in range(len(mat[0])):
                    if mat[i][j] == 'o':
                        cluster_id = len(clusters.keys())
                        clusters[cluster_id] = [(i, j)]
                        expand(cluster_id, clusters, i, j, mat, k=5, pixels=pixels)
        except RecursionError as e:
            print(f"Error with cluster search: {e}")
            return 'zero'

        # We only keep clusters that are bigger than 20 pixels volume
        size_threshold = 20
        clusters = dict(filter(lambda x: len(x[1]) >= size_threshold, clusters.items()))
        n = len(clusters.keys())
        print(f"n pixel clusters: {len(clusters.keys())}")
        if n == 0 :
            n = 'zero'
        elif n == 1:
            n = 'one'
        else:
            n = 'many'

        return n