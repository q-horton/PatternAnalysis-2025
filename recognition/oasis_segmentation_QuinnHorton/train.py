import dataset
import modules
import torch
from torchvision import transforms

device = torch.accelerator.current_accelerator().type if\
    torch.accelerator.is_available() else "cpu"

model = modules.UNet().to(device)

# See things
train_dataloader = dataset.get_dataloader(dataset.TRAIN, 16, True)
validate_dataloader = dataset.get_dataloader(dataset.VALIDATE, 16, False)
