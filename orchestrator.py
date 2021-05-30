import os
from typing import List, Tuple

from clingo.solving import Model
from asp import ASPLoader
from dataloader import load_input
import clingo

import logging
logger = logging.getLogger('orchestrator')

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from socialstructure import SocialStructure

class Orchestrator():
    def __init__(self, social_structure: SocialStructure):
        """Orchestrator class that 'orchestrates' a social structure of agents and rules.

        Args:
            social_structure (SocialStructure): object that represent a set of agents and rules.
        """
        if not social_structure:
            logger.error(f"Error: Please make sure that you have provided a social structure of agents")
            return

        self.social_structure = social_structure
        self.program = social_structure.to_ASP()
        self.answer_sets = []

    def __repr__(self):
        return (f"""
        Orchestrator.
        Number of agents: {len(self.social_structure.agents)}
        Agents: {[f"{agent.small_name}: {agent.description}" for agent in self.social_structure.agents]}
        Program: {self.program}
        """)
    
    def on_model(self, model: Model) -> None:
        """
        Callback for when we are able to retrieve a set of stable models
        """
        answer_set = []
        for atom in model.symbols(atoms="True"):
            answer_set.append(str(atom))
        self.answer_sets.append(answer_set)

    def infer(self, inputs_dict: dict) -> List[List[str]]:
        """
        Make an inference on a given set of inputs
        - First we let each individual neural network make an inference on the given inputs
        - We then parse the inferences into the ASP program
        - Finally we find a stable model under the newly parsed program
        """

        parsed_program = self.program

        # We pass the relevant inputs for each model based on their input_type so that they can make an inference
        inferences = self.social_structure.infer(inputs_dict)

        # Instantiate a Clingo solver and solve it given our parsed program
        clingo_control = clingo.Control([])
        clingo_control.add("base", [], parsed_program)
        clingo_control.ground([("base", [])])

        # We induct the inference values into the external atoms defined in the program, by setting their truth value
        for name, inference in inferences.items():
            clingo_control.assign_external(clingo.Function(f"{name}", [clingo.Function(inference)]), True)

        clingo_control.solve(on_model=self.on_model)

        return self.answer_sets

    def validate(self, directory: str, truth_label: str, n: int) -> Tuple[int]:
        """
        Function to validate a 
        """
        n_correct = 0
        n_incorrect = 0
        for image in os.listdir(directory)[:n]:
            image_path = os.path.join(directory, image)
            print(image_path)

            # Initialize our inputs dictionary and process the paths into data tensors
            inputs_dict = {"image": image_path}
            inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}

            answer_sets = self.infer(inputs_tensor_dict)

            if truth_label in answer_sets[-1]:
                n_correct += 1
            else:
                n_incorrect += 1

            print(answer_sets[-1])

        precision = round((n_correct / n)*100, 2) if n_correct else 0.00
        recall = round((n_correct / (n_correct+n_incorrect))*100, 2) if n_correct else 0.00
        accuracy = round((n_correct / n)*100, 2) if n_correct else 0.00
        f1 = round(2*((precision*recall)/(precision+recall)), 2)  if precision and recall else 0.00
        print(f"""\nValidation finished...
                    \tN total samples: {n}
                    \tN correct predictions: {n_correct}
                    \tN incorrect predictions: {n_incorrect}
                    \tRecall: {recall}%
                    \tAccuracy: {accuracy}%
                    \tF1: {f1}%""")

        return (precision, recall, accuracy, f1)

    def train():
        return ""
