import argparse
import sys
sys.path.append('asp')

from dataloader import load_input
from orchestrator import Orchestrator
from models.cnc import CNC_net
from models.hl import HL_net
from models.pc import PC_net


"""
Program to make an inference using given ASP rules and nn's
x Define program
x Load models
x Make inferences
x Parse outputs
x Retrieve SM
"""

inference_program = """
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
    parser.add_argument("-image", help="Provide the path to an input image.", required=False)
    parser.add_argument("-text", help="Provide the path to an input text.", required=False)
    parser.add_argument("-o", help="Provide the label(s) which you want the network to classify.", required=False)
    args = parser.parse_args()

    image_filename = args.image if args.image else 'example_image.tif'
    output_labels = args.o.split(',') if args.o else []
    
    print(f"Our program is defined as:\n{inference_program}")
    
    print(f"Now we load in our nn's")
    model_mapping = {"cnc": CNC_net(),
                     "hl": HL_net(),
                     "pc": PC_net()}

    # Initialize our inputs dictionary and process the paths into data tensors
    inputs_dict = {"image": image_filename}
    inputs_tensor_dict = {name:load_input(name, path) for name, path in inputs_dict.items()}

    orchestrator = Orchestrator(inference_program, model_mapping)
    print(repr(orchestrator))

    answer_sets = orchestrator.infer(inputs_tensor_dict)

    print(f"Answer sets: {answer_sets}")
    if output_labels:
        print(f"The truth value of the desired output(s):")
        for output_label in output_labels:
            print(f"{output_label}:{output_label in answer_sets[0]}")