from PIL import Image
import os
import torch
from torchvision import transforms
from dotenv import load_dotenv

# Load relevant environment variables
load_dotenv()
OASIS_PATH = os.getenv('OASIS_PATH')
TRAINING_FOLDER = os.getenv('TRAINING_FOLDER')
TESTING_FOLDER = os.getenv('TESTING_FOLDER')
VALIDATION_FOLDER = os.getenv('VALIDATION_FOLDER')

# Constant values
TEST = 0
TRAIN = 1
VALIDATE = 2


class OASISDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path):
        '''
        Loads the contents of the provided directory into an
        easily processed form.
        '''
        # Load all file names from the provided directory
        files = os.listdir(f"{img_dir_path}")

        # Initialise member variables
        self.path = img_dir_path
        self.files = files
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
            ])
        self.num_samples = len(files)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load all slices of the sample
        image = Image.open(f"{self.path}/{self.files[idx]}").convert('L')
        image = self.transform(image)

        return image, torch.tensor(idx, dtype=torch.long)


class OASISDataset_3D(torch.utils.data.Dataset):
    def __init__(self, img_dir_path):
        '''
        Load dataset metadata, assuming image name is of the form
        ???_xxx_???_yyy.???
        where xxx and yyy are the sample and slice indices
        and assuming that all samples have equal slices
        '''
        # Load all file names from the provided directory
        files = os.listdir(f"{img_dir_path}")

        # Extract all of the sample IDs in the file paths
        sample_ids = []
        for i in files:
            idx = int(i.split("_", 2)[1])
            if idx not in sample_ids:
                sample_ids.append(idx)
        sample_ids.sort()

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
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
            ])
        self.num_samples = int(len(files) / num_slices)
        self.sample_slices = num_slices
        self.naming_vars = [name_s[0], name_s[2], extension]
        self.sample_ids = sample_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load all slices of the sample
        slices = []
        id = self.sample_ids[idx]
        for i in range(self.sample_slices):
            file = f"{self.path}/{self.naming_vars[0]}_{id:03d}_" +\
                    f"{self.naming_vars[1]}_{i}.{self.naming_vars[2]}"
            image = Image.open(file).convert('L')
            image = self.transform(image)
            slices.append(image)

        sample = torch.stack(slices, 0)
        sample = torch.squeeze(sample)

        return sample, torch.tensor(id, dtype=torch.long)


def get_dataloader(data_type, batch_size):
    # Load dataset for usage
    if data_type == VALIDATE:
        files_dir = f"{OASIS_PATH}/{VALIDATION_FOLDER}"
    elif data_type == TEST:
        files_dir = f"{OASIS_PATH}/{TESTING_FOLDER}"
    else:
        files_dir = f"{OASIS_PATH}/{TRAINING_FOLDER}"
    files = OASISDataset(files_dir)

    # Create a DataLoader to assist in the batching process
    return torch.utils.data.DataLoader(files, batch_size=batch_size,
                                       shuffle=True)
