import os
import argparse 

from orchestrator import Orchestrator
from socialstructure import SocialStructure
from rule import Rule
from condition import *

from models.cnc import CNC_net
from models.hl import HL_net
from models.pc_experimental import PC_net
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

    social_structure = SocialStructure()
    cnc = CNC_net()
    hl = HL_net()
    pc = PC_net()
    social_structure.add_agent(cnc)
    social_structure.add_agent(hl)
    # social_structure.add_agent(pc)

    social_structure.add_rule(Rule("chemicalimage", [PositiveCondition(cnc, "chemical"), PositiveCondition(hl, "chemicalstructure")]))
    social_structure.add_rule(Rule("nonchemicalimage", [PositiveCondition(cnc, "nonchemical"), NegativeCondition(hl, "chemicalstructure")]))
    # social_structure.add_rule(Rule("onechemicalstructure", [LiteralCondition("chemicalimage"), PositiveCondition(pc, "one")]))
    # social_structure.add_rule(Rule("manychemicalstructure", [LiteralCondition("chemicalimage"), PositiveCondition(pc, "many")]))

    orchestrator = Orchestrator(social_structure)
    print(repr(orchestrator))

    start_time = time.time()
    precision, recall, accuracy, f1 = orchestrator.validate(directory, truth_label, n)

    end_time = time.time()
    print(f"Validation took: {round(end_time-start_time, 3)}s")