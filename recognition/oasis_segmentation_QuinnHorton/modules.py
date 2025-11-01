import torch
from torch import nn
from torchvision import transforms
import dataset

device = torch.accelerator.current_accelerator().type if\
    torch.accelerator.is_available() else "cpu"

# 3D Conv given: nn.Conv3d
# Context Module - ?
# Stride 2 Conv given: nn.Conv3d
# Upsampling - Only 2d given
# Localisation - ?
# Segmentation - ?
# Softmax - Only for 2d
# Elementwise sum - ?
# Concatenation - ?
# Upscaling - ?


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = UNet().to(device)
print(model)

# See things
dataloader = dataset.get_dataloader(dataset.TEST, 16)
for images, labels in dataloader:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image shape: {images[0][0].shape}")
    picture = transforms.functional.to_pil_image(images[0][0])
    picture.show()
    break
