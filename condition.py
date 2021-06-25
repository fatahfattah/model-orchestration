

class Condition:
    def __init__(self) -> None:
        pass

    def to_ASP():
        return ""

class LiteralCondition(Condition):
    def __init__(self, literal) -> None:
        """Literal condition of form 'a.'.

        Args:
            literal (String): E.g. 'a'
        """
        super().__init__()
        self.literal = literal

    def to_ASP(self):
        return f"{self.literal}"

class PositiveCondition(Condition):
    def __init__(self, classifier, outcome) -> None:
        """A positive condition with for an classifier output 'model(a)'

        Args:
            classifier (String): modelname
            outcome (String): output label
        """
        super().__init__()
        self.classifier = classifier
        self.outcome = outcome
    
    def to_ASP(self):
        return f"{self.classifier.small_name}({self.outcome})"

class NegativeCondition(Condition):
    def __init__(self, classifier, outcome) -> None:
        """A negative condition with for an classifier output e.g. ' not model(a)'

        Args:
            classifier (String): modelname
            outcome (String): output label
        """
        super().__init__()
        self.classifier = classifier
        self.outcome = outcome
    
    def to_ASP(self):
        return f"not {self.classifier.small_name}({self.outcome})"

class ComparisonCondition(Condition):
    def __init__(self, classifier, outcome, operator, value) -> None:
        """A comparison condition on an classifier's inference e.g. 'model(a) > 1'

        Args:
            classifier (String): modelname
            outcome (String): output label
            operater (String): operator such as >, <, >=, <=, ==
            value (String): comparison value
        """
        super().__init__()
        self.classifier = classifier
        self.outcome = outcome
        self.operator = operator
        self.value = value

    def to_ASP(self):
        return f"{self.classifier.small_name}({self.outcome}) {self.operator} {self.value}"
