

import torch
import torchvision.transforms as transforms

class Agent:
    """
    Parent class of an Agent
    """

    def __init__(self, classes, name, small_name, description, input_type, inference_type):
        self.classes = classes
        self.exploration_classes = classes + [f"not {c}" for c in classes]
        self.name = name
        self.small_name = small_name
        self.description = description
        self.input_type = input_type
        self.inference_type = inference_type

    def to_ASP(self) -> str:
        """
        Convert agent to ASP representation.

        RETURNS: ASP representation
        """
        return f"#external {self.small_name}({';'.join(self.classes)})."

    def infer(self, input, explore=False) -> str:
        """
        Make an inference for this agent. 

        RETURNS: Agent inference
        """
        return ""