import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import os

# --- CIFAR10 数据加载函数 ---
def load_cifar10_data(batch_size=64):
    """
    加载CIFAR-10数据集，用于分类任务。
    返回的DataLoader会提供 (图像, 标签) 数据对。
    """
    transform = transforms.Compose([
        transforms.ToTensor() # 将Pillow图像转换为Tensor，并自动将像素值从[0, 255]归一化到[0, 1]
    ])
    
    # 使用原始的CIFAR10类，默认返回 (图像, 标签)
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

class KodakDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        return image_tensor, image_tensor

def load_kodak_dataset(path, batch_size=1):
    transform = transforms.Compose([transforms.ToTensor()])
    kodak_dataset = KodakDataset(root_dir=path, transform=transform)
    data_loader = DataLoader(kodak_dataset, batch_size=batch_size, shuffle=False)
    return data_loader