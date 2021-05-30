
class SocialStructure:
    def __init__(self) -> None:
        """
        A social structure that consists of resources (agents) and rules.
        """
        self.agents = []
        self.rules = []

    def add_agent(self, agent):
        """
        Add a new agent to the social structure.
        Every agent has to be an 'Agent' inherited class type.

        Args:
            agent (Agent): Agent to be added.
        """
        self.agents.append(agent)

    def add_rule(self, rule):
        """
        Add a new rule to the social structure ruleset.
        Current support for rules with and without a head, e.g 'a :- b' or ':- b'.

        Args:
            rule (Rule): Rule to be added.
        """
        self.rules.append(rule)

    def infer(self, inputs_dict):
        """Run inferences on all agents for the current input:

        Args:
            inputs_dict (dict): e.g. {'image':[[]]}

        Returns:
            {model_name:inference}: Key value pair dicht with model_names as keys and their inference as value
        """
        return {agent.small_name:agent.infer(inputs_dict[agent.input_type]) for agent in self.agents}

    def to_ASP(self):
        """Parse the social structure to its respective ASP representation.

        Returns:
            String: ASP representation of the agents and rules.
        """
        program = ""
        program += "\n".join([agent.to_ASP() for agent in self.agents]) # E.g. model(a,b,c)
        program += "\n"
        program += "\n".join([rule.to_ASP() for rule in self.rules])
        return program