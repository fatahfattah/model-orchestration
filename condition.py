

class Condition:
    def __init__(self) -> None:
        pass

    def to_ASP():
        return ""

class ConstantCondition(Condition):
    def __init__(self, constant) -> None:
        """Constant condition of form 'a.'.

        Args:
            constant (String): E.g. 'a'
        """
        super().__init__()
        self.constant = constant

    def to_ASP(self):
        return f"{self.constant}"

class FunctionCondition(Condition):
    def __init__(self, classifier, outcome) -> None:
        """A function condition for an classifier output 'model(a)'

        Args:
            classifier (String): modelname
            outcome (String): output label
        """
        super().__init__()
        self.classifier = classifier
        self.outcome = outcome
    
    def to_ASP(self):
        return f"{self.classifier.small_name}({self.outcome})"

class NegativeFunctionCondition(Condition):
    def __init__(self, classifier, outcome) -> None:
        """A negative function condition for an classifier output e.g. ' not model(a)'

        Args:
            classifier (String): modelname
            outcome (String): output label
        """
        super().__init__()
        self.classifier = classifier
        self.outcome = outcome
    
    def to_ASP(self):
        return f"not {self.classifier.small_name}({self.outcome})"
