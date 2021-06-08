import argparse
from socialstructure import SocialStructure
import sys
sys.path.append('asp')

from dataloader import load_input

from socialstructure import SocialStructure
from condition import *
from rule import *

from orchestrator import Orchestrator
from models.cnc import CNC_net
from models.cncmany import CNCMANY_net
from models.hl import HL_net
from models.pc_experimental import PC_net


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
    parser.add_argument("-o", help="Provide the label(s) which you want the network to classify.", required=False)
    args = parser.parse_args()

    image_filename = args.image if args.image else 'example_image.tif'
    output_labels = args.o.split(',') if args.o else []
    
    social_structure = SocialStructure()
    
    cnc = CNC_net()
    cncmany = CNCMANY_net()
    hl = HL_net()
    pc = PC_net()

    social_structure.add_agent(cnc)
    social_structure.add_agent(cncmany)
    social_structure.add_agent(hl)
    social_structure.add_agent(pc)

    # social_structure.add_rule(Rule("onechemicalstructure", [PositiveCondition(cncmany, "onechemical")]))
    # social_structure.add_rule(Rule("manychemicalstructure", [PositiveCondition(cncmany, "manychemical")]))
    # social_structure.add_rule(Rule("nonchemicalimage", [PositiveCondition(cncmany, "nonchemical")]))
    social_structure.add_rule(Rule("chemicalimage", [PositiveCondition(cnc, "onechemical"), PositiveCondition(hl, "chemicalstructure")]))
    social_structure.add_rule(Rule("nonchemicalimage", [PositiveCondition(cnc, "nonchemical"), NegativeCondition(hl, "chemicalstructure")]))
    social_structure.add_rule(Rule("onechemicalstructure", [LiteralCondition("chemicalimage"), PositiveCondition(pc, "one")]))
    social_structure.add_rule(Rule("manychemicalstructure", [LiteralCondition("chemicalimage"), PositiveCondition(pc, "many")]))

    # Initialize our inputs dictionary and process the paths into data tensors
    inputs_dict = {"image": image_filename}
    inputs_tensor_dict = {name:load_input(name, path) for name, path in inputs_dict.items()}

    orchestrator = Orchestrator(social_structure)
    print(repr(orchestrator))

    answer_sets = orchestrator.infer(inputs_tensor_dict)

    print(f"Answer sets: {answer_sets}")
    if output_labels:
        print(f"The truth value of the desired output(s):")
        for output_label in output_labels:
            print(f"{output_label}:{output_label in answer_sets[-1]}")