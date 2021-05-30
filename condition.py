

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
    def __init__(self, agent, outcome) -> None:
        """A positive condition with for an agent output 'model(a)'

        Args:
            agent (String): modelname
            outcome (String): output label
        """
        super().__init__()
        self.agent = agent
        self.outcome = outcome
    
    def to_ASP(self):
        return f"{self.agent.small_name}({self.outcome})"

class NegativeCondition(Condition):
    def __init__(self, agent, outcome) -> None:
        """A negative condition with for an agent output e.g. ' not model(a)'

        Args:
            agent (String): modelname
            outcome (String): output label
        """
        super().__init__()
        self.agent = agent
        self.outcome = outcome
    
    def to_ASP(self):
        return f"not {self.agent.small_name}({self.outcome})"

class ComparisonCondition(Condition):
    def __init__(self, agent, outcome, operator, value) -> None:
        """A comparison condition on an agent's inference e.g. 'model(a) > 1'

        Args:
            agent (String): modelname
            outcome (String): output label
            operater (String): operator such as >, <, >=, <=, ==
            value (String): comparison value
        """
        super().__init__()
        self.agent = agent
        self.outcome = outcome
        self.operator = operator
        self.value = value

    def to_ASP(self):
        return f"{self.agent.small_name}({self.outcome}) {self.operator} {self.value}"
