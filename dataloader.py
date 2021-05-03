import torch
import numpy as np

from PIL import Image

def load_input(input_type, input_path):
    if input_type == "image":
        return Image.open(input_path)
