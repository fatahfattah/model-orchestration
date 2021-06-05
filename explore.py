import os
import random
random.seed(42)
import argparse

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataloader import load_input
from models.cnc import CNC_net
from models.cncmany import CNCMANY_net
from models.hl import HL_net
from models.pc_experimental import PC_net
"""
We use the orchestrator to 'explore' our dataset, trying to find relations between model outputs.

Example:
We have trained model A, on ...
We have ancillery model b

Let model a and model b infer all samples from dataset
We can now create relationship matrix between inferences from model a and model b

Steps:
- Choose a dataset;
- Run inference for pre-trained models
- Run inferences for aucillery models, on same input.
- Create confusion matrix that represents relationships for models to aucillery models.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-root_dir", help="Please provide the path to the root directory wherein directory names are truth labels.")
    parser.add_argument("-n", help="Please provide the number of samples that you want to explore, otherwise all will be validated", required=False)
    args = parser.parse_args()

    root_dir = args.root_dir if args.root_dir else ""
    n = int(args.n) if args.n else float('inf')

    print(f"We will start model exploration...\nDataset directory: {root_dir}")

    print(f"Now we load in our nn's")
    model_mapping = {
                     "cnc": CNC_net(),
                    #  "cncmany": CNCMANY_net(),
                     "hl": HL_net(),
                     "pc": PC_net()
                     }

    truth_labels = os.listdir(root_dir)
    inferences = {}
    for truth_label in truth_labels:
        directory = os.path.join(root_dir, truth_label)

        for image in random.sample(os.listdir(directory), min(n, len(os.listdir(directory)))):
            image_path = os.path.join(directory, image)
            print(image_path)

            # Initialize our inputs dictionary and process the paths into data tensors
            inputs_dict = {"image": image_path}
            inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}

            for name, agent in model_mapping.items():
                inferences.setdefault(name, {})
                inferences[name].setdefault(truth_label, []).append(agent.infer(inputs_tensor_dict[agent.input_type]))
    

    fig = plt.figure()
    fig_index = 1
    filters = {}
    for name, agent in model_mapping.items():
        agent_inferences = inferences[name]
        classes = agent.classes
        y_headers = list(agent_inferences.keys())
        mat = matrelative = [[0 for i in range(len(classes))] for i in range(len(y_headers))]
        
        for i, label in enumerate(y_headers):
            for pred in agent_inferences[label]:
                mat[i][classes.index(pred)] += 1

        matsums = [sum(row) for row in mat]
        for i, row in enumerate(mat):
            for j, cell in enumerate(row):
                if cell:
                    matrelative[i][j] = round(cell/matsums[i], 2)
                    if matrelative[i][j] >= 0.1:
                        filters.setdefault(name, {})
                        filters[name].setdefault(y_headers[i], []).append(classes[j])

        ax = fig.add_subplot(len(model_mapping.keys()), 1, fig_index)
        ax.set_title(agent.name)
        sns.heatmap(matrelative, annot=True)
        ax.set_yticklabels(y_headers, rotation = 360, va = 'center')
        ax.set_xticklabels(classes, rotation = -45)
        fig_index += 1
    
    print(f"We have found the following possible filters: {filters}")
    fig.tight_layout()
    plt.show()
