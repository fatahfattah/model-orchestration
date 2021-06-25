from classifier import Classifier
from rule import Rule

class SocialStructure:
    def __init__(self) -> None:
        """
        A social structure that consists of resources (classifiers) and rules.
        """
        self.classifiers = []
        self.rules = []

    def add_classifier(self, classifier: Classifier) -> None:
        """
        Add a new classifier to the social structure.
        Every classifier has to be an 'Classifier' inherited class type.

        Args:
            classifier (Classifier): Classifier to be added.
        """
        self.classifiers.append(classifier)

    def add_rule(self, rule: Rule) -> None:
        """
        Add a new rule to the social structure ruleset.
        Current support for rules with and without a head, e.g 'a :- b' or ':- b'.

        Args:
            rule (Rule): Rule to be added.
        """
        self.rules.append(rule)

    def infer(self, inputs_dict: dict, explore=False) -> dict:
        """Run inferences on all classifiers for the current input:

        Args:
            inputs_dict (dict): e.g. {'image':[[]]}

        Returns:
            {model_name:inference}: Key value pair dict with model_names as keys and their inference as value
        """
        if explore:
            return {classifier.small_name:inference for classifier in self.classifiers for inference in classifier.infer(inputs_dict[classifier.input_type])}

        return {classifier.small_name:classifier.infer(inputs_dict[classifier.input_type]) for classifier in self.classifiers}

    def to_ASP(self) -> str:
        """Parse the social structure to its respective ASP representation.

        Returns:
            String: ASP representation of the classifierss and rules.
        """
        program = ""
        program += "\n".join([classifier.to_ASP() for classifier in self.classifiers]) # E.g. model(a,b,c)
        program += "\n"
        program += "\n".join([rule.to_ASP() for rule in self.rules])
        return program