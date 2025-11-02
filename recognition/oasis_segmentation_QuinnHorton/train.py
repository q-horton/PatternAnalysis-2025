import dataset
import modules
import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if\
    torch.accelerator.is_available() else "cpu"


def save_model(model):
    torch.save(model.state_dict(), modules.PATH)


def train_model():
    # Dataloaders
    train_dataloader = dataset.get_dataloader(dataset.TRAIN, 1, True)
    validate_dataloader = dataset.get_dataloader(dataset.VALIDATE, 1, False)

    # Load the NN model
    model = modules.UNet().to(device)

    # Use the Dice Loss Criterion in-line with the specification
    criterion = nn.CrossEntropyLoss()  # modules.DiceLoss()

    # Use the adam optimiser per Isensee
    optimiser = torch.optim.Adam(model.parameters())

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            masks = torch.squeeze(masks, dim=1)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimiser.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Validation phase (conceptual)
        model.eval()
        with torch.no_grad():
            dsc_vals = []
            for images, masks in validate_dataloader:
                masks = torch.squeeze(masks, dim=1)
                for i in range(images.shape[0]):
                    dsc_vals.append(modules.DSC(images, masks))
            dsc_val = sum(dsc_vals) / len(dsc_vals)
            print(f"Average Dice Similarity Coefficient: {dsc_val}\n---")

    save_model(model)


train_model()
