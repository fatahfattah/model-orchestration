from asp import ASPLoader
import clingo

import logging

logger = logging.getLogger('orchestrator')

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
        This is an orchestrator instance.
        Number of models: {len(self.models.keys())}
        Models: {[f"{name}: {model.description}" for name, model in self.models.items()]}
        """)
    
    def on_model(self, model):
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

        # Replace our neural rules with the actual inferences
        for model_name, inference in inferences.items():
            parsed_program = "\n".join([f"{model_name}({inference})." 
                                        if line.startswith("nn(") and model_name in line 
                                        else line for line in parsed_program.split('\n')])
        
        clingo_control = clingo.Control([])
        clingo_control.add("base", [], parsed_program)
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model=self.on_model)

        return self.answer_sets