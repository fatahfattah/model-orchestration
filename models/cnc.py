import torch
import torchvision.transforms as transforms

from PIL import Image

import numpy as np

classes = ['chemical', 'nonchemical']

class CNC_net():
    """
    Chemical/non-chemical (CNC) network
    Predict whether an image contains a chemical structure or not.
    """

    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
        self.model.load_state_dict(torch.load('./models/cnc.pth', map_location=torch.device('cpu')))
        self.model.eval()
        self.description = "Classifies whether an image contains a chemical structure depiction"
        self.input_size = (299,299)
        self.input_type = "image"
        self.inference_type = "classification"
        self.base_model = 'inception_v3'
        self.preprocessing = transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Resize(self.input_size),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])

    def infer(self, image):
        """
        Parse image
        Infer
        Return
        """
        image = image.convert('RGB')
        image = self.preprocessing(image)
        # image = image.unsqueeze(0)
        with torch.no_grad():
            self.model(image)

        return "chemical"