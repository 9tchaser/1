# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
random.seed(42)
#定义超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr=0.0002
batch_size = 128
num_epoch = 50

lr_image_path=r"D:\MONAI\CGAN_Image_sup\uct_pair_data_dachaung\uct_pair_data_dachaung/tx8/train"
hr_image_path=r"D:\MONAI\CGAN_Image_sup\uct_pair_data_dachaung\uct_pair_data_dachaung/tx512/train"

# 图像转换操作
transform = transforms.Compose([
    transforms.Resize((64)),  # 调整图片大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 转换为Tensor并进行归一化
])

# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, lr_image_path, hr_image_path, transform=None):
        self.lr_image_path = lr_image_path
        self.hr_image_path = hr_image_path
        self.transform = transform
        self.lr_images = os.listdir(lr_image_path)
        self.hr_images = os.listdir(lr_image_path)

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        lr_image_path = os.path.join(self.lr_image_path, self.lr_images[index])
        lr_image = Image.open(lr_image_path).convert('L')
        lr_image = self.transform(lr_image)
        hr_image_path = os.path.join(self.hr_image_path, self.hr_images[index])
        hr_image = Image.open(hr_image_path).convert('L')
        hr_image = self.transform(hr_image)
      
        return lr_image,hr_image

# 创建自定义Dataset实例
train_dataset = CustomDataset(lr_image_path, hr_image_path, transform=transform)
# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %%
#定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# %%
#定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# %%
#网络初始化
generator=Generator().to(device)
discriminator=Discriminator().to(device)
#定义损失函数
loss= nn.BCELoss()
g_optimizer=optim.Adam(generator.parameters(),lr=lr)
d_optimizer=optim.Adam(discriminator.parameters(),lr=lr)

# %%
#训练
for epoch in range(num_epoch):
    for i, (lr_images, hr_images) in enumerate(train_loader):
            lr_images = lr_images.cuda()
            hr_images = hr_images.cuda()

            # 训练判别器
            d_optimizer.zero_grad()
            fake_hr_images = generator(lr_images)
            fake_pair = torch.cat((lr_images, fake_hr_images), dim=1)
            real_pair = torch.cat((lr_images, hr_images), dim=1)
            output_fake = discriminator(fake_pair.detach())
            output_real = discriminator(real_pair)
            loss_D = loss(output_fake, torch.zeros_like(output_fake)) + loss(output_real, torch.ones_like(output_real))
            loss_D.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_pair)
            loss_G = loss(output_fake, torch.ones_like(output_fake))
            loss_G.backward()
            g_optimizer.step()

            # 打印损失
            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch+1, epoch, i+1, len(train_loader), loss_D.item(), loss_G.item()))
with torch.no_grad():
    samples = generator(lr_images).cpu().data
    figs,axs=plt.subplots(1,10,figsize=(10,1))
    for j in range(10):
        axs[j].imshow(samples[j].reshape(64,64),cmap='gray')
        axs[j].axis('off')
    plt.show()