

class Condition:
    def __init__(self) -> None:
        pass

    def to_ASP():
        return ""

class LiteralCondition(Condition):
    def __init__(self, literal) -> None:
        super().__init__()
        self.literal = literal

    def to_ASP(self):
        return f"{self.literal}"

class PositiveCondition(Condition):
    def __init__(self, agent, outcome) -> None:
        super().__init__()
        self.agent = agent
        self.outcome = outcome
    
    def to_ASP(self):
        return f"{self.agent.small_name}({self.outcome})"

class NegativeCondition(Condition):
    def __init__(self, agent, outcome) -> None:
        super().__init__()
        self.agent = agent
        self.outcome = outcome
    
    def to_ASP(self):
        return f"not {self.agent.small_name}({self.outcome})"

class ComparisonCondition(Condition):
    def __init__(self, agent, outcome, operator, value) -> None:
        super().__init__()
        self.agent = agent
        self.outcome = outcome
        self.operator = operator
        self.value = value

    def to_ASP(self):
        return f"{self.agent.small_name}({self.outcome}) {self.operator} {self.value}"
