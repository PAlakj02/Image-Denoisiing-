import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the paths to the directories containing the low-resolution and predicted images
low_res_dir =.\test\low'
predicted_dir = .\test\predicted'

# Define the transform to convert PIL images to tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])
# Define a subset size for testing
subset_size = 10  # Test on first 10 images or batches
# Define PSNR calculation function
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 10 * torch.log10(1 / mse)

class Generator(nn.Module):
    def __init__(self, c_dim=3):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(c_dim, 64, kernel_size=4, stride=2, padding=1)     # (128, 128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)       # (64, 64)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)      # (32, 32)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)      # (16, 16)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)      # (8, 8)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)      # (4, 4)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)      # (2, 2)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)      # (1, 1)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1) # (2, 2)
        self.deconv2 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1) # (4, 4)
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1) # (8, 8)
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1) # (16, 16)
        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # (32, 32)
        self.deconv6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # (64, 64)
        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (128, 128)
        self.deconv8 = nn.ConvTranspose2d(64, c_dim, kernel_size=4, stride=2, padding=1) # (256, 256)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x))
        
        
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv4(x))
        x = torch.relu(self.deconv5(x))
        x = torch.relu(self.deconv6(x))
        x = torch.tanh(self.deconv7(x))  
        x = torch.tanh(self.deconv8(x))
        return x
    

# Initialize Generator and load pre-trained weights (if available)
generator = Generator()
generator.load_state_dict(torch.load(generator_full.pth', map_location=torch.device('cpu')))
generator.eval()

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {image_file}: {str(e)}")
                return None

        return image

# Create the dataset and dataloader for testing
test_dataset = ImageDataset(low_res_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Iterate through the dataset and process images
predicted_tensors = []
for idx, img in enumerate(test_dataloader):
    with torch.no_grad():
        pred_img = generator(img)
    predicted_tensors.append(pred_img)
    # Save predicted image to disk
    pred_img = pred_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    pred_img = (pred_img + 1) / 2.0
    pred_img = np.clip(pred_img, 0, 1)
    pred_img = Image.fromarray((pred_img * 255).astype(np.uint8))
    pred_img.save(os.path.join(predicted_dir, f'predicted_{idx}.png'))

# Optionally, calculate PSNR if you have ground truth high-resolution images
# Replace 'ground_truth_dir' with your actual directory containing high-resolution images
ground_truth_dir = .\Train\high'
ground_truth_dataset = ImageDataset(ground_truth_dir, transform=transform)
ground_truth_dataloader = DataLoader(ground_truth_dataset, batch_size=1, shuffle=False)

psnr_values = []
for pred_tensor, gt_img in zip(predicted_tensors, ground_truth_dataloader):
   psnr = calculate_psnr(pred_tensor, gt_img)
psnr_values.append(psnr.item())

print("PSNR values:")
print(psnr_values)
