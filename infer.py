import sys
sys.path.append('..')

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from orchestrator import Orchestrator

from models.cnc import CNC_net
from models.hl import HL_net

from tqdm import tqdm

"""
Program to make an inference using given ASP rules and nn's
x Define program
- Load models
- Make inferences
- Parse outputs
- Retrieve SM
"""

inference_program = """
% We define our neural rules
nn(cnc, image, [chemical, nonchemical])
nn(hl, image, ['character', 'chemicalstructure', 'drawing', 'flowchart', 'genesequence', 'graph', 'math', 'programlisting', 'table'])

% If both cnc and hl infer chemical, the image is chemical
chemical_image :- cnc(chemical), hl(chemicalstructure)

% If either cnc or hl infer non chemical, the image is not chemical
non_chemical_image :- cnc(nonchemical)
non_chemical_image :- not hl(chemicalstructure)
"""

if __name__ == "__main__":
    print(f"Our program is defined as:\n{inference_program}")
    print(f"Now we load in our nn's")
    cnc = CNC_net()
    hl = HL_net()

    model_mapping = {"cnc":cnc,
                     "hl": hl}

    orchestrator = Orchestrator(inference_program, model_mapping)
    print(repr(orchestrator))


