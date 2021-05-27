import os
import random
random.seed(42)
import argparse

import matplotlib.pyplot as plt

from dataloader import load_input
from models.cnc import CNC_net
from models.hl import HL_net
from models.pc import PC_net
"""
We use the orchestrator to 'explore' our dataset, trying to find relations between model outputs.

Example:
We have trained model A, on ...
We have aucillery model b

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
    n = int(args.n) if args.n else 10

    print(f"We will start model exploration...\nDataset directory: {root_dir}")

    print(f"Now we load in our nn's")
    model_mapping = {
                    #  "cnc": CNC_net(),
                     "hl": HL_net(),
                    #  "pc": PC_net()
                     }

    inferences = {}
    for truth_label in os.listdir(root_dir):
        directory = os.path.join(root_dir, truth_label)

        for image in random.sample(os.listdir(directory), min(n, len(os.listdir(directory)))):
            image_path = os.path.join(directory, image)

            # Initialize our inputs dictionary and process the paths into data tensors
            inputs_dict = {"image": image_path}
            inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}
            inferences.setdefault(truth_label, []).append({name:model.infer(inputs_tensor_dict[model.input_type]) for name, model in model_mapping.items()})
    
    n_labels = len(inferences.keys())
    fig = plt.figure()
    fig_index = 1
    for i, v in inferences.items():
        print(f"truth: {i} ->")
        n = len(v)
        cooccurances = {}

        headers = []
        mat = []
        for inference in v:
            cooccur_key = "+".join([x for x in inference.values()])
            cooccurances.setdefault(cooccur_key, 0)
            cooccurances[cooccur_key] += 1
            
        cooccurances = dict(sorted(cooccurances.items(), key=lambda x:x[1], reverse=True))
        for cooccur_key, cooccur_n in cooccurances.items():
            headers.append(cooccur_key)
            mat.append(cooccur_n)
            print(f"\t{cooccur_key}".ljust(20), f"n: {cooccur_n},".ljust(5), f"{round(cooccur_n/n*100, 2)}%")

        # We need a 2d mat to plot
        mat = [mat]
        ax = fig.add_subplot(n_labels, 1, fig_index)
        img = plt.imshow(mat, interpolation=None)
        ax.set_xticks([i for i in range(len(headers))])
        ax.set_xticklabels(headers, rotation=-45)
        ax.set_yticks([0])
        ax.set_yticklabels([i])
        plt.colorbar(img)
        print("\n")
        fig_index+=1
    
    fig.tight_layout()
    plt.show()

    

