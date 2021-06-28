import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAdjustSharpness
from classifier import Classifier

from orchestrator import Orchestrator
from socialstructure import SocialStructure
from rule import Rule
from condition import *

from models.drawing import DRAWING_net
from models.not_drawing import NOTDRAWING_net
from models.cncmany_drawfilter_positive import CNCMANY_DRAWFILTER_POSITIVE_net
from models.cncmany_drawfilter_negative import CNCMANY_DRAWFILTER_NEGATIVE_net

class CNCMANY_AGGREGATED_net(Classifier):
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
        positive_classifier = CNCMANY_DRAWFILTER_POSITIVE_net()
        negative_classifier = CNCMANY_DRAWFILTER_NEGATIVE_net()
        drawing = DRAWING_net()
        not_drawing = NOTDRAWING_net()

        social_structure.add_classifier(positive_classifier)
        social_structure.add_classifier(negative_classifier)
        social_structure.add_classifier(drawing)
        social_structure.add_classifier(not_drawing)

        # Add our filtering rules
        social_structure.add_rule(Rule("filter_many", [PositiveCondition(drawing, "drawing")]))
        social_structure.add_rule(Rule("filter_none", [PositiveCondition(not_drawing, "not_not_drawing")]))
        social_structure.add_rule(Rule("filter_one", [PositiveCondition(not_drawing, "not_drawing")]))

        social_structure.add_rule(Rule("positive_manychemical", [PositiveCondition(positive_classifier, "manychemical"), LiteralCondition("filter_many")]))
        social_structure.add_rule(Rule("positive_nonchemical", [PositiveCondition(positive_classifier, "nonchemical"), LiteralCondition("filter_none")]))
        social_structure.add_rule(Rule("positive_onechemical", [PositiveCondition(positive_classifier, "onechemical"), LiteralCondition("filter_one")]))

        social_structure.add_rule(Rule("negative_manychemical", [PositiveCondition(negative_classifier, "manychemical")
                                                                , LiteralCondition("not positive_manychemical")
                                                                , LiteralCondition("not positive_nonchemical")
                                                                , LiteralCondition("not positive_onechemical")]))

        social_structure.add_rule(Rule("negative_nonchemical", [PositiveCondition(negative_classifier, "nonchemical")
                                                                , LiteralCondition("not positive_manychemical")
                                                                , LiteralCondition("not positive_nonchemical")
                                                                , LiteralCondition("not positive_onechemical")]))

        social_structure.add_rule(Rule("negative_onechemical", [PositiveCondition(negative_classifier, "onechemical")
                                                                , LiteralCondition("not positive_manychemical")
                                                                , LiteralCondition("not positive_nonchemical")
                                                                , LiteralCondition("not positive_onechemical")]))

        social_structure.add_rule(Rule("manychemical", [LiteralCondition("positive_manychemical")]))
        social_structure.add_rule(Rule("manychemical", [LiteralCondition("negative_manychemical")]))

        social_structure.add_rule(Rule("nonchemical", [LiteralCondition("positive_nonchemical")]))
        social_structure.add_rule(Rule("nonchemical", [LiteralCondition("negative_nonchemical")]))

        social_structure.add_rule(Rule("onechemical", [LiteralCondition("positive_onechemical")]))
        social_structure.add_rule(Rule("onechemical", [LiteralCondition("negative_onechemical")]))

        self.orchestrator = Orchestrator(social_structure)


    def infer(self, image):
        """
        Parse image
        Infer
        Return
        """

        return self.orchestrator.infer({'image':image})[-1][-1]