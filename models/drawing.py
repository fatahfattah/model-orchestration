import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAdjustSharpness
from classifier import Classifier

class DRAWING_NN(Classifier):
    """
    Drawing/not drawing network
    Predict whether an image has depicted a drawing.
    """

    def __init__(self):
        super().__init__(['drawing', 'not_drawing'],
                        "Drawing/not drawing network",
                        "drawing",
                        "Predict whether an image has depicted a drawing.",
                        "image",
                        "classification")

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
        self.model.fc = torch.nn.Linear(2048, len(self.classes))
        self.model.load_state_dict(torch.load('./models/drawing.pth', map_location=torch.device('cpu')))
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