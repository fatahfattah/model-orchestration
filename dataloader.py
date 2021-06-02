from PIL import Image
from torchvision.datasets import ImageFolder


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # Return (mat, class) + path
        return (super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],))

def load_input(input_type, input_path):
    """
    Load an image according to their input type.
    """
    if input_type == "image":
        return Image.open(input_path)
    elif input_type == "text":
        with open(input_path, "r") as f:
            return f.read()
    