import os
from typing import List, Tuple

from matplotlib.pyplot import draw 
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from clingo.solving import Model
from asp import ASPLoader
from dataloader import load_input
import clingo
import time

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

    def validate(self, directory: str, n: int, visualize = True: bool) -> Tuple[int]:
        """
        Function to validate a 
        """
        start_time = time.time()
        classes = os.listdir(directory)
        mat = [[0 for i in range(len(classes))] for i in range(len(classes))]
        outs = []
        true_labels = []    
        for truth_label in classes:
            images = os.listdir(os.path.join(directory, truth_label))
            
            for image in images[:min(n, len(images))]:
                image_path = os.path.join(directory, truth_label, image)

                # Initialize our inputs dictionary and process the paths into data tensors
                inputs_dict = {"image": image_path}
                inputs_tensor_dict = {name: load_input(name, path) for name, path in inputs_dict.items()}
                answer_sets = self.infer(inputs_tensor_dict)
                output = answer_sets[-1][-1]
                print(f"truth: {truth_label}, prediction: {output}, image: {image}")
                if output in classes:
                    true_labels.append(truth_label)
                    outs.append(output)
                    mat[classes.index(truth_label)][classes.index(output)] += 1

        end_time = time.time()
        print(f"Validation took: {round(end_time-start_time, 3)}s")
        print(classification_report(true_labels, outs))

        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(f"Confusion matrix of validation {classes}")
            sns.heatmap(mat, annot=True)
            ax.set_yticklabels([l for l in classes], 
                                rotation = 360, 
                                va = 'center')
                                
            ax.set_xticklabels(classes, rotation = -45)

            plt.show()



    def train():
        return ""
