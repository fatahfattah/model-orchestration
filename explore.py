import os
import random
random.seed(42)
import argparse

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataloader import load_input
from models.cnc import CNC_NN
from models.cncmany import CNCMANY_NN
from models.hl import HL_NN
from models.drawing import DRAWING_NN
from models.not_drawing import NOTDRAWING_NN
from models.pc_experimental import PC_NN

from ranking import Ranking, rank_grouped_rankings

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
                    # "cnc": CNCMANY_NN()
                    #  "hl": HL_NN(),
                     "drawing": DRAWING_NN(),
                     "not_drawing": NOTDRAWING_NN(),
                    #  "pc": PC_NN()
                     }

    target_classifier = CNCMANY_NN()

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

            # If the target classifier makes a wrong prediction, we will register the ancillery outputs
            target_classifier_inference = target_classifier.infer(inputs_tensor_dict[target_classifier.input_type])
            for name, classifier in model_mapping.items():
                # Register inferences for current ancillery classifier
                inferences.setdefault(name, {})
                classifier_inferences = classifier.infer(inputs_tensor_dict[classifier.input_type])
                # Register each inference for this true label
                for inf in classifier_inferences if isinstance(classifier_inferences, list) else [classifier_inferences]:
                    if target_classifier_inference != truth_label:
                        inferences[name].setdefault(f"(wrong) {truth_label}", []).append(inf)
                    else:
                        inferences[name].setdefault(f"(correct) {truth_label}", []).append(inf)

    fig = plt.figure()
    fig_index = 1
    filters = {}
    rankings = []
    for name, classifier in model_mapping.items():
        if not name in inferences:
            continue

        classifier_inferences = inferences[name]
        classes = classifier.classes
        y_headers = list(classifier_inferences.keys())
        mat = [[0 for i in range(len(classes))] for i in range(len(y_headers))]
        matrelative = [[0 for i in range(len(classes))] for i in range(len(y_headers))]

        # Register the amount of hits for each true_label (correct/wrong) combination with ancillery output
        for i, label in enumerate(y_headers):
            for pred in classifier_inferences[label]:
                mat[i][classes.index(pred)] += 1

        matsums = [sum(row) for row in mat]
        for i, row in enumerate(mat):
            for j, cell in enumerate(row):
                if cell:
                    matrelative[i][j] = round(cell/matsums[i], 2)
                    if matrelative[i][j] >= 0.1:
                        filters.setdefault(name, {})
                        filters[name].setdefault(y_headers[i], []).append(classes[j])

        for i, c in enumerate(classes):
            for t in truth_labels:
                wrong_i = y_headers.index(f"(wrong) {t}") if f"(wrong) {t}" in y_headers else None
                wrong_ratio = matrelative[wrong_i][i] if wrong_i != None else 0
                wrong_n = mat[wrong_i][i] if wrong_i != None else 0
                correct_i = y_headers.index(f"(correct) {t}") if f"(correct) {t}" in y_headers else None
                correct_ratio = matrelative[correct_i][i] if correct_i != None else 0
                correct_n = mat[correct_i][i] if correct_i != None else 0
                rankings.append(Ranking(t, name, c, wrong_n, correct_n, wrong_ratio, correct_ratio, wrong_ratio-correct_ratio))

        ax = fig.add_subplot(len(model_mapping.keys()), 1, fig_index)
        ax.set_title(classifier.name)
        sns.heatmap(matrelative, annot=True)
        ax.set_yticklabels([l for l in y_headers], 
                            rotation = 360, 
                            va = 'center')
                            
        ax.set_xticklabels(classes, rotation = -45)
        fig_index += 1
    
    fig.tight_layout()

    fig = plt.figure()
    fig_index = 1
    for t in truth_labels:
        grouped_rankings = [r for r in rankings if r.target_label == t]
        rank_grouped_rankings(grouped_rankings)
        print(f"Relations for {t} -> ")
        for r in grouped_rankings:
            print(r)
        
        ax = fig.add_subplot(len(truth_labels), 1, fig_index)
        ax.set_title(f"Rankings for: {t}")
        y_headers = [f"{r.ancillery_classifier}({r.ancillery_label})" for r in grouped_rankings]
        rankings_mat = [r.to_mat() for r in grouped_rankings]
        sns.heatmap(rankings_mat, annot=True)
        ax.set_yticklabels([l for l in y_headers], rotation=360, va='center')
        ax.set_xticklabels([l for l in ['correct_n', 'wrong_n', 
                                        'correct_rank', 'wrong_rank', 'wrong_minus_correct_rank', 
                                        'wc_ranking_rank',  'relative_n_rank', 'relative_rank_rank']],
                                        rotation=-15)
        fig_index += 1

    fig.tight_layout()
    plt.show()
