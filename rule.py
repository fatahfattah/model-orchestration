
class Rule:
    def __init__(self, output, conditions) -> None:
        self.output = output
        self.conditions = conditions
    
    def to_ASP(self) -> str:
        return f"{self.output} :- {', '.join([condition.to_ASP() for condition in self.conditions])}."

