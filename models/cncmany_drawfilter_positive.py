import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAdjustSharpness
from classifier import Classifier

class CNCMANY_DRAWFILTER_POSITIVE_NN(Classifier):
    """
    Chemical/non-chemical/many-chemical network specialized with drawingfilter (CNC)(positive) network
    Predict whether an image contains a chemical structure or not.
    """

    def __init__(self):
        super().__init__(['manychemical', 'nonchemical', 'onechemical'],
                        "Chemical/non-chemical/many-chemical network specialized with drawingfilter (positive)",
                        "cncmany_drawfilter_positive",
                        "Classifies whether an image contains a chemical structure depiction",
                        "image",
                        "classification")

        
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
        self.model.fc = torch.nn.Linear(2048, len(self.classes))
        try:
            self.model.load_state_dict(torch.load('./models/cncmany_drawingfilter_relative_rank.pth', map_location=torch.device('cpu')))
        except FileNotFoundError as e:
            print(f"ERROR: make sure that the model exists if you want to use {self.small_name} for inferences")
        self.model.eval()
        self.input_size = (299,299)
        self.base_model = 'inception_v3'
        self.inference_preprocessing = transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Resize(self.input_size),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])
        self.preprocessing = transforms.Compose([
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])
        self.train_preprocessing = transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomAdjustSharpness(0),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ])


    def infer(self, image):
        """
        Parse image
        Infer
        Return
        """
        prediction = ""

        image = image.convert('RGB')
        image = self.inference_preprocessing(image)
        with torch.no_grad():
            outputs = self.model(image[None, ...])
            aux, predicted = torch.max(outputs, 1)
            confidence = round(aux[0].item(), 2)
            prediction = self.classes[predicted[0]]

        print(f"{self.small_name}: {prediction}@{confidence}")
        return prediction