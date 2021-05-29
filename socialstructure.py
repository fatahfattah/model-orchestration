
class SocialStructure:
    """
    A social structure that consists of resources (agents) and rules.
    """

    def __init__(self) -> None:
        self.agents = []
        self.rules = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_rule(self, rule):
        self.rules.append(rule)

    def infer(self, inputs_dict):
        return {agent.small_name:agent.infer(inputs_dict[agent.input_type]) for agent in self.agents}

    def to_ASP(self):
        program = ""
        program += "\n".join([agent.to_ASP() for agent in self.agents]) # E.g. model(a,b,c)
        program += "\n"
        program += "\n".join([rule.to_ASP() for rule in self.rules])
        return program