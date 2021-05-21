
import os
import argparse
from dataloader import load_input
from orchestrator import Orchestrator
from models.cnc import CNC_net
from models.hl import HL_net
from models.pc import PC_net



"""
We validate our network given a set of datapoints with truth label
E.g. Validate directory_a which contains chemicals

Output: N datapoints, N correct, N incorrect, etc.
"""

validation_program = """
% We define our neural rules
#external cnc(chemical;nonchemical).
#external hl(character;chemicalstructure;drawing;flowchart;genesequence;graph;math;programlisting;table).
#external pc(n_clusters).

% If both cnc and hl infer chemical, the image is chemical
chemicalimage :- cnc(chemical), hl(chemicalstructure).

% If either cnc or hl infer non chemical, the image is not chemical
nonchemicalimage :- cnc(nonchemical).
nonchemicalimage :- not hl(chemicalstructure).

% If there is a chemicalimage and multiple pixel clusters, we have one chemical depiction
onechemicalstructure :- chemicalimage, pc(n_clusters) == 1.

% If there is a chemicalimage and multiple pixel clusters, we have many chemical depiction
manychemicalstructure :- chemicalimage, pc(n_clusters) > 1.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-directory", help="Please provide the path to the directory that contains the validation data.")
    parser.add_argument("-truthlabel", help="Please provide the truth label for this dataset.", )
    parser.add_argument("-n", help="Please provide the number of samples that you want to validate, otherwise we will validate all", required=False)
    args = parser.parse_args()

    directory = args.directory if args.directory else ""    
    truth_label = args.truthlabel if args.truthlabel else ""
    n = int(args.n) if args.n else len(os.listdir(directory))

    print(f"We will start validation...\n\tValidation for: {truth_label}\n\tDirectory: {directory}\n\tN samples: {n}")

    print(f"Our program is defined as:\n{validation_program}")
    
    print(f"Now we load in our nn's")
    model_mapping = {"cnc": CNC_net(),
                     "hl": HL_net(),
                     "pc": PC_net()}

    orchestrator = Orchestrator(validation_program, model_mapping)
    print(repr(orchestrator))

    n_correct = 0
    n_incorrect = 0
    for image in os.listdir(directory)[:n]:
        image_path = os.path.join(directory, image)
        print(image_path)

        # Initialize our inputs dictionary and process the paths into data tensors
        inputs_dict = {"image": image_path}
        inputs_tensor_dict = {name:load_input(name, path) for name, path in inputs_dict.items()}

        answer_sets = orchestrator.infer(inputs_tensor_dict)

        if truth_label in answer_sets[-1]:
            n_correct += 1
        else:
            n_incorrect += 1

        print(answer_sets)

    precision = round((n_correct / n)*100, 2)
    recall = round((n_correct / (n_correct+n_incorrect))*100, 2) if n_correct else 0.00
    accuracy = round((n_correct / n)*100, 2) if n_correct else 0.00
    f1 = round(2*((precision*recall)/(precision+recall)), 2)
    print(f"""\nValidation finished...
                \tN total samples: {n}
                \tN correct predictions: {n_correct}
                \tN incorrect predictions: {n_incorrect}
                \tRecall: {recall}%
                \tAccuracy: {accuracy}%
                \tF1: {f1}%""")
                