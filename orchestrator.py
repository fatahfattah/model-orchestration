

class Orchestrator():
    def __init__(self, program, models):
        self.program = program
        self.models = models

    def __repr__(self):
        return f"""
        This is an orchestrator instance.
        Number of models: {len(self.models.keys())}
        Models: {[f"{name}: {model.description}" for name, model in self.models.items()]}
        ASP program: {self.program}
        """