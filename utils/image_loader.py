import pytesseract
from PIL import Image

def load_image(file):

    img = Image.open(file)

    return pytesseract.image_to_string(img)