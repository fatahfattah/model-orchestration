import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAdjustSharpness
from classifier import Classifier

from orchestrator import Orchestrator
from socialstructure import SocialStructure
from rule import Rule
from condition import *

from models.cncmany import CNCMANY_NN
from models.drawing import DRAWING_NN
from models.not_drawing import NOTDRAWING_NN
from models.cncmany_drawfilter_positive import CNCMANY_DRAWFILTER_POSITIVE_NN
from models.cncmany_drawfilter_negative import CNCMANY_DRAWFILTER_NEGATIVE_NN

class CNCMANY_AGGREGATED_NN(Classifier):
    """
    Aggregated Chemical/non-chemical/Many-chemical (CNC) network
    Predict whether an image contains a chemical structure or not.
    """

    def __init__(self):
        super().__init__(['manychemical', 'nonchemical', 'onechemical'],
                        "Aggregated Chemical/non-chemical/many-chemical network",
                        "cncmany_aggregated",
                        "Classifies whether an image contains a chemical structure depiction, using a joint-decision structure",
                        "image",
                        "classification")

        social_structure = SocialStructure()
        cncmany = CNCMANY_NN()
        positive_classifier = CNCMANY_DRAWFILTER_POSITIVE_NN()
        negative_classifier = CNCMANY_DRAWFILTER_NEGATIVE_NN()
        drawing = DRAWING_NN()
        not_drawing = NOTDRAWING_NN()

        social_structure.add_classifier(cncmany)
        social_structure.add_classifier(positive_classifier)
        social_structure.add_classifier(negative_classifier)
        social_structure.add_classifier(drawing)
        social_structure.add_classifier(not_drawing)

        # Add our filtering rules
        social_structure.add_rule(Rule("filter_many", [FunctionCondition(drawing, "drawing")]))
        social_structure.add_rule(Rule("filter_none", [FunctionCondition(not_drawing, "not_not_drawing")]))
        social_structure.add_rule(Rule("filter_one", [FunctionCondition(not_drawing, "not_drawing")]))

        social_structure.add_rule(Rule("manychemical", [FunctionCondition(positive_classifier, "manychemical"), ConstantCondition("filter_many")]))
        social_structure.add_rule(Rule("nonchemical", [FunctionCondition(positive_classifier, "nonchemical"), ConstantCondition("filter_none")]))
        social_structure.add_rule(Rule("onechemical", [FunctionCondition(positive_classifier, "onechemical"), ConstantCondition("filter_one")]))

        social_structure.add_rule(Rule("manychemical", [FunctionCondition(negative_classifier, "manychemical")
                                                                , ConstantCondition("not filter_many")]))

        social_structure.add_rule(Rule("nonchemical", [FunctionCondition(negative_classifier, "nonchemical")
                                                                , ConstantCondition("not filter_none")]))

        social_structure.add_rule(Rule("onechemical", [FunctionCondition(negative_classifier, "onechemical")
                                                                , ConstantCondition("not filter_one")]))

        social_structure.add_rule(Rule("manychemical", [FunctionCondition(cncmany, "manychemical")
                                                                , ConstantCondition("not onechemical")
                                                                , ConstantCondition("not nonchemical")]))
                                                                
        social_structure.add_rule(Rule("nonchemical", [FunctionCondition(cncmany, "nonchemical")
                                                                , ConstantCondition("not manychemical")
                                                                , ConstantCondition("not onechemical")]))

        social_structure.add_rule(Rule("onechemical", [FunctionCondition(cncmany, "onechemical")
                                                                , ConstantCondition("not manychemical")
                                                                , ConstantCondition("not nonchemical")]))



        self.orchestrator = Orchestrator(social_structure)


    def infer(self, image):
        """
        Parse image
        Infer
        Return
        """
        return self.orchestrator.infer({'image':image})[-1][-1]