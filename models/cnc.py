import torch
import sys, os

classes = ['chemical', 'nonchemical']

class CNC_net():
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
        self.model.load_state_dict(torch.load('./models/cnc.pth'))
        self.description = "Classifies whether an image contains a chemical structure depiction"
        self.input_size = (299,299)
        self.input_type = "image"
        self.inference_type = "classification"
        self.base_model = 'inception_v3'


    def infer(self, image):
        """
        Parse image
        Infer
        Return
        """

        return "nonchemical"