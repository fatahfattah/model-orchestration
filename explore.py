import os
import random
random.seed(42)
import argparse
from dataloader import load_input

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
    model_mapping = {"cnc": CNC_net(),
                     "hl": HL_net(),
                    #  "pc": PC_net()
                     }

    inferences = {}
    for truth_label in os.listdir(root_dir):
        directory = os.path.join(root_dir, truth_label)

        for image in random.sample(os.listdir(directory), n):
            image_path = os.path.join(directory, image)

            # Initialize our inputs dictionary and process the paths into data tensors
            inputs_dict = {"image": image_path}
            inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}
            inferences.setdefault(truth_label, []).append({name:model.infer(inputs_tensor_dict[model.input_type]) for name, model in model_mapping.items()})
    
    for i, v in inferences.items():
        print(f"truth: {i} ->")
        for inference in v:
            print(f"\t{inference}")
        print("\n")

    confusion = {}
    for truth, _inferences in inferences.items():
        confusion.setdefault(truth, {})
        for v in _inferences:
            for _, inference in v.items():
                confusion[truth].setdefault(inference, 0)
                confusion[truth][inference] += 1

    for truth, confusions in confusion.items():
        print(f"truth: {truth} confused with ->")
        for k, v in confusions.items():
            print(f"\t{k}: {v}")

        print("\n")

    