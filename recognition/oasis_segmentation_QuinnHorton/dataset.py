from PIL import Image
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
OASIS_PATH = os.getenv('OASIS_PATH')

training_files = os.listdir(f"{OASIS_PATH}/keras_png_slices_seg_train")
image = Image.open(f"{OASIS_PATH}/keras_png_slices_seg_train/{training_files[0]}")

print(image.format)
print(image.size)
print(image.mode)

list = np.array([image])

image.show()
