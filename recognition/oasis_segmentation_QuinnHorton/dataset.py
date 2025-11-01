from PIL import Image
import os
import torch
from torchvision import datasets, transforms
from dotenv import load_dotenv

# Load relevant environment variables
load_dotenv()
OASIS_PATH = os.getenv('OASIS_PATH')

# Load dataset for usage
transform = transforms.Compose([
    transforms.ToTensor()
    ])
training_files_dir = f"{OASIS_PATH}"  # /keras_png_slices_seg_train"
training_files = datasets.ImageFolder(training_files_dir, transform=transform)

# Create a DataLoader to assist in the batching process
dataloader = torch.utils.data.DataLoader(training_files, batch_size=32,
                                         shuffle=True)

# See things
for images, labels in dataloader:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image shape: {images[0][0].shape}")
    picture = transforms.functional.to_pil_image(images[0][0])
    picture.show()
    print(labels[0])
    break
