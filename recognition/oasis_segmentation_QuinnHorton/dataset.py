from PIL import Image
import os
import torch
from torchvision import datasets, transforms
from dotenv import load_dotenv

# Load relevant environment variables
load_dotenv()
OASIS_PATH = os.getenv('OASIS_PATH')


class OASISDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path, transform=None):
        '''
        Load dataset metadata, assuming image name is of the form
        ???_xxx_???_yyy.???
        where xxx and yyy are the sample and slice indices
        and assuming that all samples have equal slices
        '''
        # Load all file names from the provided directory
        files = os.listdir(f"{img_dir_path}")

        # Process file names to establish data size
        prototype, extension = files[0].split(".", 1)
        name_s = prototype.split("_")
        num_slices = 0
        while True:
            name_s[3] = str(num_slices)
            if f"{'_'.join(name_s)}.{extension}" not in files:
                break
            num_slices += 1

        # Initialise member variables
        self.path = img_dir_path
        self.transform = transform
        self.num_samples = int(len(files) / num_slices)
        self.sample_slices = num_slices
        self.naming_vars = [name_s[0], name_s[2], extension]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load image (example for a common image format, adjust for NIfTI)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


ods = OASISDataset(f"{OASIS_PATH}/keras_png_slices_train")
print(f"Length of dataset: {len(ods)}")

# # Load dataset for usage
# transform = transforms.Compose([
#     transforms.ToTensor()
#     ])
# training_files_dir = f"{OASIS_PATH}/keras_png_slices_seg_train"
# training_files = datasets.DatasetFolder(training_files_dir, transform=transform)
# 
# # Create a DataLoader to assist in the batching process
# dataloader = torch.utils.data.DataLoader(training_files, batch_size=32,
#                                          shuffle=True)
# 
# # See things
# for images, labels in dataloader:
#     print(f"Image batch shape: {images.shape}")
#     print(f"Label batch shape: {labels.shape}")
#     print(f"Image shape: {images[0][0].shape}")
#     picture = transforms.functional.to_pil_image(images[0][0])
#     picture.show()
#     break
