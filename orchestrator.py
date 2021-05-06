from asp import ASPLoader
import clingo

import logging
logger = logging.getLogger('orchestrator')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class Orchestrator():
    def __init__(self, program, models):
        if not program or not models:
            logger.error(f"Error: Please make sure that you have provided a ASP program and the relevant models")
            return

        self.program = program
        self.models = models
        self.answer_sets = []

    def __repr__(self):
        return (f"""
        Orchestrator.
        Number of models: {len(self.models.keys())}
        Models: {[f"{name}: {model.description}" for name, model in self.models.items()]}
        """)
    
    def on_model(self, model):
        """
        Callback for when we are able to retrieve a set of stable models
        """
        answer_set = []
        for atom in model.symbols(atoms="True"):
            answer_set.append(str(atom))
        self.answer_sets.append(answer_set)

    def infer(self, inputs_dict):
        """
        Make an inference on a given set of inputs
        - First we let each individual neural network make an inference on the given inputs
        - We then parse the inferences into the ASP program
        - Finally we find a stable model under the newly parsed program
        """

        parsed_program = self.program

        # We pass the relevant inputs for each model based on their input_type so that they can make an inference
        inferences = {name:model.infer(inputs_dict[model.input_type]) for name, model in self.models.items()}

        # Instantiate a Clingo solver and solve it given our parsed program
        clingo_control = clingo.Control([])
        clingo_control.add("base", [], parsed_program)
        clingo_control.ground([("base", [])])

        # We induct the inference values into the external atoms defined in the program, by setting their truth value
        for name, inference in inferences.items():
            clingo_control.assign_external(clingo.Function(f"{name}", [clingo.Function(inference)]), True)

        clingo_control.solve(on_model=self.on_model)

        return self.answer_sets

    def train():
        return ""

    def validate():
        return ""
