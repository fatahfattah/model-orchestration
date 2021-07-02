from models.cncmany import CNCMANY_net
import os
import argparse

from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from dataloader import load_input

from orchestrator import Orchestrator
from socialstructure import SocialStructure
from rule import Rule
from condition import *

from models.cncmany_drawfilter_positive import CNCMANY_DRAWFILTER_POSITIVE_net
from models.cncmany_drawfilter_negative import CNCMANY_DRAWFILTER_NEGATIVE_net
from models.cncmany_aggregated import CNCMANY_AGGREGATED_net
from models.drawing import DRAWING_net
from models.not_drawing import NOTDRAWING_net
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
    parser.add_argument("-n", help="Please provide the number of samples that you want to validate, otherwise we will validate all", required=False)
    args = parser.parse_args()

    directory = args.directory if args.directory else ""
    n = int(args.n) if args.n else float('inf')

    print(f"We will start validation...\n\tDirectory: {directory}")

    social_structure = SocialStructure()
    cncmany = CNCMANY_net()
    cncmany_aggregated = CNCMANY_AGGREGATED_net()
    cncmany_drawfilter_positive = CNCMANY_DRAWFILTER_POSITIVE_net()
    cncmany_drawfilter_negative = CNCMANY_DRAWFILTER_NEGATIVE_net()
    drawing = DRAWING_net()
    not_drawing = NOTDRAWING_net()

    social_structure.add_classifier(cncmany)
    social_structure.add_classifier(cncmany_aggregated)
    social_structure.add_classifier(cncmany_drawfilter_positive)
    social_structure.add_classifier(cncmany_drawfilter_negative)
    social_structure.add_classifier(drawing)
    social_structure.add_classifier(not_drawing)

    social_structure.add_rule(Rule("onechemical", [FunctionCondition(cncmany_aggregated, "onechemical")]))
    social_structure.add_rule(Rule("manychemical", [FunctionCondition(cncmany_aggregated, "manychemical")]))
    social_structure.add_rule(Rule("nonchemical", [FunctionCondition(cncmany_aggregated, "nonchemical")]))

    orchestrator = Orchestrator(social_structure)
    print(repr(orchestrator))

    orchestrator.validate(directory, n)