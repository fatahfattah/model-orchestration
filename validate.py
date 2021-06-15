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

"""
#external drawing(drawing;not_drawing).
#external not_drawing(not_drawing;not_not_drawing).
#external cncmany_drawfilter_positive(manychemical;nonchemical;onechemical);
#external cncmany_drawfilter_negative(manychemical;nonchemical;onechemical);

manychemical_drawfilter :- drawing(drawing).
nonchemical_drawfilter :- not_drawing(not_not_drawing).
onechemical_drawfilter :- not_drawing(not_drawing).

manychemical :- cncmany_drawfilter_positive(manychemical), manychemical_drawfilter.
nonchemical :- cncmany_drawfilter_positive(nonchemical), nonchemical_drawfilter.
onechemical :- cncmany_drawfilter_positive(onechemical), onechemical_drawfilter.

manychemical :- cncmany_drawfilter_negative(manychemical), not manychemical_drawfilter.
nonchemical :- cncmany_drawfilter_negative(nonchemical), not nonchemical_drawfilter.
onechemical :- cncmany_drawfilter_negative(onechemical), not onechemical_drawfilter.
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
    cncmany_drawfilter_positive = CNCMANY_DRAWFILTER_POSITIVE_net()
    cncmany_drawfilter_negative = CNCMANY_DRAWFILTER_NEGATIVE_net()
    drawing = DRAWING_net()
    not_drawing = NOTDRAWING_net()

    social_structure.add_agent(cncmany_drawfilter_positive)
    social_structure.add_agent(cncmany_drawfilter_negative)
    social_structure.add_agent(drawing)
    social_structure.add_agent(not_drawing)

    # Add our filtering rules
    social_structure.add_rule(Rule("manychemical_drawfilter", [PositiveCondition(drawing, "drawing")]))
    social_structure.add_rule(Rule("nonchemical_drawfilter", [PositiveCondition(not_drawing, "not_not_drawing")]))
    social_structure.add_rule(Rule("onechemical_drawfilter", [PositiveCondition(not_drawing, "not_drawing")]))

    social_structure.add_rule(Rule("manychemical", [PositiveCondition(cncmany_drawfilter_positive, "manychemical"), LiteralCondition("manychemical_drawfilter")]))
    social_structure.add_rule(Rule("nonchemical", [PositiveCondition(cncmany_drawfilter_positive, "nonchemical"), LiteralCondition("nonchemical_drawfilter")]))
    social_structure.add_rule(Rule("onechemical", [PositiveCondition(cncmany_drawfilter_positive, "onechemical"), LiteralCondition("onechemical_drawfilter")]))

    social_structure.add_rule(Rule("manychemical", [PositiveCondition(cncmany_drawfilter_negative, "manychemical"), LiteralCondition("not manychemical_drawfilter")]))
    social_structure.add_rule(Rule("nonchemical", [PositiveCondition(cncmany_drawfilter_negative, "nonchemical"), LiteralCondition("not nonchemical_drawfilter")]))
    social_structure.add_rule(Rule("onechemical", [PositiveCondition(cncmany_drawfilter_negative, "onechemical"), LiteralCondition("not onechemical_drawfilter")]))


    orchestrator = Orchestrator(social_structure)
    print(repr(orchestrator))

    # start_time = time.time()
    # precision, recall, accuracy, f1 = orchestrator.validate(directory, truth_label, n)

    # end_time = time.time()
    # print(f"Validation took: {round(end_time-start_time, 3)}s")

    """
    Give a directory
    outs = []
    labels = []
    """

    classes = os.listdir(directory)
    mat = [[0 for i in range(len(classes))] for i in range(len(classes))]
    outs = []
    true_labels = []
    for truth_label in classes:
        for image in os.listdir(os.path.join(directory, truth_label))[:n]:
            image_path = os.path.join(directory, truth_label, image)

            # Initialize our inputs dictionary and process the paths into data tensors
            inputs_dict = {"image": image_path}
            inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}
            answer_sets = orchestrator.infer(inputs_tensor_dict)
            true_labels.append(truth_label)
            output = answer_sets[-1][-1]
            print(f"truth: {truth_label}, prediction: {output}, image: {image}")
            outs.append(output)
            if output in classes:
                mat[classes.index(truth_label)][classes.index(output)] += 1


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