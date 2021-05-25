import os
import argparse 
from orchestrator import Orchestrator
from models.cnc import CNC_net
from models.hl import HL_net
from models.pc import PC_net
import time


"""
We validate our network given a set of datapoints with truth label
E.g. Validate directory_a which contains chemicals

Output: N datapoints, N correct, N incorrect, etc.

Alternatively, we provide a path to a dataset and we take directory names as truth.
E.g.; Validation_dataset
            - Chemical
            - Nonchemical
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
    parser.add_argument("-truth_label", help="Please provide the truth label for this dataset.", )
    parser.add_argument("-n", help="Please provide the number of samples that you want to validate, otherwise we will validate all", required=False)
    args = parser.parse_args()

    directory = args.directory if args.directory else ""    
    truth_label = args.truth_label if args.truth_label else ""
    n = int(args.n) if args.n else len(os.listdir(directory))

    print(f"We will start validation...\n\tValidation for: {truth_label}\n\tDirectory: {directory}\n\tN samples: {n}")

    print(f"Our program is defined as:\n{validation_program}")

    print(f"Now we load in our nn's")
    model_mapping = {"cnc": CNC_net(),
                     "hl": HL_net(),
                     "pc": PC_net()}

    orchestrator = Orchestrator(validation_program, model_mapping)
    print(repr(orchestrator))

    start_time = time.time()
    precision, recall, accuracy, f1 = orchestrator.validate(directory, truth_label, n)

    end_time = time.time()
    print(f"Validation took: {round(end_time-start_time, 3)}s")