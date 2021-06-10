import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomAdjustSharpness
from agent import Agent

class NOTDRAWING_net(Agent):
    """
    (Negative) Not drawing/ not not drawing network
    Predict whether an image has not depicted a drawing.
    """

    def __init__(self):
        super().__init__(['not_drawing', 'not_not_drawing'],
                        "(Negative) Not drawing/ not not drawing network",
                        "drawing",
                        "Predict whether an image has not depicted a drawing.",
                        "image",
                        "classification")

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
        self.model.fc = torch.nn.Linear(2048, len(self.classes))
        self.model.load_state_dict(torch.load('./models/not_drawing.pth', map_location=torch.device('cpu')))
        self.model.eval()
        self.input_size = (299,299)
        self.base_model = 'inception_v3'
        self.preprocessing = transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Resize(self.input_size),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])
        self.train_preprocessing = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize(self.input_size),
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
        image = self.preprocessing(image)
        with torch.no_grad():
            outputs = self.model(image[None, ...])
            aux, predicted = torch.max(outputs, 1)
            confidence = round(aux[0].item(), 2)
            prediction = self.classes[predicted[0]]

        print(f"{self.small_name}: {prediction}@{confidence}")
        return prediction