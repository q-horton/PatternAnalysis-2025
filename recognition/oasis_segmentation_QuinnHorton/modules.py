from torchvision import transforms
import dataset


# See things
dataloader = dataset.get_dataloader(dataset.TEST, 16)
for images, labels in dataloader:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image shape: {images[0][0].shape}")
    picture = transforms.functional.to_pil_image(images[0][0])
    picture.show()
    break
