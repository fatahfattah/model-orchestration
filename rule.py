
class Rule:
    def __init__(self, outheadput, body) -> None:
        self.head = head
        self.body = body
    
    def to_ASP(self) -> str:
        return f"{self.head} :- {', '.join([condition.to_ASP() for condition in self.body])}."

