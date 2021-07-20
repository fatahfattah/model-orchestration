import argparse
from socialstructure import SocialStructure
import sys
sys.path.append('asp')

from dataloader import load_input

from socialstructure import SocialStructure
from condition import *
from rule import *

from orchestrator import Orchestrator
from models.cnc import CNC_NN
from models.cncmany_aggregated import CNCMANY_AGGREGATED_NN
from models.cncmany import CNCMANY_NN
from models.hl import HL_NN
from models.pc_experimental import PC_NN


"""
Program to make an inference using given ASP rules and nn's
x Define program
x Load models
x Make inferences
x Parse outputs
x Retrieve stable model
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-image", help="Provide the path to an input image.", required=False)
    parser.add_argument("-text", help="Provide the path to an input text.", required=False)
    args = parser.parse_args()

    image_filename = args.image if args.image else 'example_image.tif'
    
    social_structure = SocialStructure()
    
    # cnc = CNC_NN()
    # cncmany_aggregated = CNCMANY_AGGREGATED_NN()
    cncmany = CNCMANY_NN()
    # hl = HL_NN()
    # pc = PC_NN()

    # social_structure.add_classifier(cnc)
    # social_structure.add_classifier(cncmany_aggregated)
    social_structure.add_classifier(cncmany)
    # social_structure.add_classifier(hl)
    # social_structure.add_classifier(pc)

    social_structure.add_rule(Rule("onechemicalstructure", [FunctionCondition(cncmany, "onechemical")]))
    social_structure.add_rule(Rule("manychemicalstructure", [FunctionCondition(cncmany, "manychemical")]))
    social_structure.add_rule(Rule("nonchemicalimage", [FunctionCondition(cncmany, "nonchemical")]))

    # Initialize our inputs dictionary and process the paths into data tensors
    inputs_dict = {"image": image_filename}
    inputs_tensor_dict = {name:load_input(name, path) for name, path in inputs_dict.items()}

    orchestrator = Orchestrator(social_structure)
    print(repr(orchestrator))

    answer_sets = orchestrator.infer(inputs_tensor_dict)

    print(f"Answer sets: {answer_sets}")
    print(f"Prediction: {answer_sets[-1][-1]}")