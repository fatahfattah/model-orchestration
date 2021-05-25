from PIL import Image

def load_input(input_type, input_path):
    """
    Load an image according to their input type.
    """
    if input_type == "image":
        return Image.open(input_path)
    elif input_type == "text":
        with open(input_path, "r") as f:
            return f.read()
    