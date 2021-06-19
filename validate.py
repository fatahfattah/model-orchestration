from models.cncmany import CNCMANY_net
import os
import argparse

from matplotlib.pyplot import draw 
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
    cncmany_drawfilter_positive = CNCMANY_DRAWFILTER_POSITIVE_net()
    cncmany_drawfilter_negative = CNCMANY_DRAWFILTER_NEGATIVE_net()
    drawing = DRAWING_net()
    not_drawing = NOTDRAWING_net()

    social_structure.add_agent(cncmany)
    social_structure.add_agent(cncmany_drawfilter_positive)
    social_structure.add_agent(cncmany_drawfilter_negative)
    social_structure.add_agent(drawing)
    social_structure.add_agent(not_drawing)

    # Add our filtering rules
    social_structure.add_rule(Rule("filter_many", [PositiveCondition(drawing, "drawing")]))
    social_structure.add_rule(Rule("filter_none", [PositiveCondition(not_drawing, "not_not_drawing")]))
    social_structure.add_rule(Rule("filter_one", [PositiveCondition(not_drawing, "not_drawing")]))

    social_structure.add_rule(Rule("positive_manychemical", [PositiveCondition(cncmany_drawfilter_positive, "manychemical"), LiteralCondition("filter_many")]))
    social_structure.add_rule(Rule("positive_nonchemical", [PositiveCondition(cncmany_drawfilter_positive, "nonchemical"), LiteralCondition("filter_none")]))
    social_structure.add_rule(Rule("positive_onechemical", [PositiveCondition(cncmany_drawfilter_positive, "onechemical"), LiteralCondition("filter_one")]))

    social_structure.add_rule(Rule("negative_manychemical", [PositiveCondition(cncmany_drawfilter_negative, "manychemical")
                                                            , LiteralCondition("not positive_manychemical")
                                                            , LiteralCondition("not positive_nonchemical")
                                                            , LiteralCondition("not positive_onechemical")]))

    social_structure.add_rule(Rule("negative_nonchemical", [PositiveCondition(cncmany_drawfilter_negative, "nonchemical")
                                                            , LiteralCondition("not positive_manychemical")
                                                            , LiteralCondition("not positive_nonchemical")
                                                            , LiteralCondition("not positive_onechemical")]))

    social_structure.add_rule(Rule("negative_onechemical", [PositiveCondition(cncmany_drawfilter_negative, "onechemical")
                                                            , LiteralCondition("not positive_manychemical")
                                                            , LiteralCondition("not positive_nonchemical")
                                                            , LiteralCondition("not positive_onechemical")]))

    social_structure.add_rule(Rule("manychemical", [LiteralCondition("positive_manychemical")]))
    social_structure.add_rule(Rule("manychemical", [LiteralCondition("negative_manychemical")]))

    social_structure.add_rule(Rule("nonchemical", [LiteralCondition("positive_nonchemical")]))
    social_structure.add_rule(Rule("nonchemical", [LiteralCondition("negative_nonchemical")]))

    social_structure.add_rule(Rule("onechemical", [LiteralCondition("positive_onechemical")]))
    social_structure.add_rule(Rule("onechemical", [LiteralCondition("negative_onechemical")]))



    orchestrator = Orchestrator(social_structure)
    print(repr(orchestrator))

    start_time = time.time()

    classes = os.listdir(directory)
    mat = [[0 for i in range(len(classes))] for i in range(len(classes))]
    outs = []
    true_labels = []
    for truth_label in classes:
        images = os.listdir(os.path.join(directory, truth_label))
        
        for image in images[:min(n, len(images))]:
            image_path = os.path.join(directory, truth_label, image)

            # Initialize our inputs dictionary and process the paths into data tensors
            inputs_dict = {"image": image_path}
            inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}
            answer_sets = orchestrator.infer(inputs_tensor_dict)
            output = answer_sets[-1][-1]
            print(f"truth: {truth_label}, prediction: {output}, image: {image}")
            if output in classes:
                true_labels.append(truth_label)
                outs.append(output)
                mat[classes.index(truth_label)][classes.index(output)] += 1

    end_time = time.time()
    print(f"Validation took: {round(end_time-start_time, 3)}s")
    print(classification_report(true_labels, outs))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(f"Confusion matrix of validation {classes}")
    sns.heatmap(mat, annot=True)
    ax.set_yticklabels([l for l in classes], 
                        rotation = 360, 
                        va = 'center')
                        
    ax.set_xticklabels(classes, rotation = -45)

    plt.show()