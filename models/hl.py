import torch
import torchvision.transforms as transforms

classes = ['character', 'chemicalstructure', 'drawing', 'flowchart', 'genesequence', 'graph', 'math', 'programlisting', 'table']

class HL_net():
    """
    Higher level (HL) network
    Classifies an image into nine classes that describe an image on a high level.
    """

    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
        # self.model.load_state_dict(torch.load('./models/hl.pth'))
        self.description = "Classifies a patent image into nine higher level image types."
        self.input_size = (299,299)
        self.input_type = "image"
        self.inference_type = "classification"
        self.base_model = 'inception_v3'
        self.preprocessing = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize((299,299)),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def infer(self, image):
        """
        Parse image
        Infer
        Return
        """

        return "chemicalstructure"