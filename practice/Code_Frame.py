import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import numpy as np
import pandas as pd
import ast
import cv2

#读取图像数据
#define .xlsx filename and folder containing images and masks
info_filename = r'D:\MONAI\CGAN_Image_sup\CGAN_Image\data\BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx'
images_and_masks_foldername = r'D:\MONAI\CGAN_Image_sup\CGAN_Image\data\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/'

#read .xlsx file with clinical data
breast_dataset = pd.read_excel(info_filename, sheet_name='BrEaST-Lesions-USG clinical dat')

for i in breast_dataset.index:
    #parse milti-label columns
    breast_dataset.at[i, 'Tissue_composition'] = breast_dataset.loc[i,'Tissue_composition'].split('&')
    breast_dataset.at[i, 'Signs'] = breast_dataset.loc[i,'Signs'].split('&')
    breast_dataset.at[i, 'Symptoms'] = breast_dataset.loc[i,'Symptoms'].split('&')
    breast_dataset.at[i, 'Margin'] = breast_dataset.loc[i,'Margin'].split('&')
    breast_dataset.at[i, 'Interpretation'] = breast_dataset.loc[i,'Interpretation'].split('&')
    breast_dataset.at[i, 'Diagnosis'] = breast_dataset.loc[i,'Diagnosis'].split('&')

    #read image
    breast_dataset.at[i, 'Image_filename']  = cv2.imread(images_and_masks_foldername+breast_dataset.loc[i,'Image_filename'], cv2.IMREAD_UNCHANGED)
    
    #read tumor mask
    if not isinstance(breast_dataset.loc[i,'Mask_tumor_filename'], float):
        mask = cv2.imread(images_and_masks_foldername+breast_dataset.loc[i,'Mask_tumor_filename'], cv2.IMREAD_GRAYSCALE)>0
        breast_dataset.at[i, 'Mask_tumor_filename']  = mask
    else:
        breast_dataset.at[i, 'Mask_tumor_filename'] = []

    #read other mask
    if not isinstance(breast_dataset.loc[i,'Mask_other_filename'], float):
        masks_bool = []
        for mask_path in breast_dataset.loc[i,'Mask_other_filename'].split('&'):
            masks_bool.append(cv2.imread(images_and_masks_foldername+mask_path, cv2.IMREAD_GRAYSCALE)>0)
        breast_dataset.at[i, 'Mask_other_filename'] = masks_bool
    else:
        breast_dataset.at[i, 'Mask_other_filename'] = []

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# 定义生成器和判别器的网络结构
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 网络结构定义，可以根据需要进行修改
        self.model = nn.Sequential(
            nn.Linear(100 + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 生成器前向传播
        gen_input = torch.cat((z, labels), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 网络结构定义，可以根据需要进行修改
        self.model = nn.Sequential(
            nn.Linear(img_shape[0] * img_shape[1] * img_shape[2] + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 判别器前向传播
        d_in = torch.cat((img.view(img.size(0), -1), labels), -1)
        validity = self.model(d_in)
        return validity

# 定义数据集类
# 定义数据集类
class MedicalDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_size=(64, 64)):
        self.dataframe = dataframe
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 从数据帧中获取图像和标签
        img_path = str(self.dataframe.iloc[idx]['Image_filename'])
        label = self.dataframe.iloc[idx]['Mask_tumor_filename']

        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # 将图像从BGR格式转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调整图像大小
        img = cv2.resize(img, self.target_size)

        # 如果需要的话，对图像进行转换
        if self.transform:
            img = self.transform(img)

        return img, label

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(), # 将图像转换为张量
    # 如果还需要其他转换，可以继续添加
])

# 设定随机种子以便复现
torch.manual_seed(42)

# 设定训练参数
epochs = 100
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
num_classes = 10 # 假设有10个不同的诊断类别
img_shape = (3, 64, 64) # 假设图像大小为64x64，并且是RGB图像

# 加载数据
medical_dataset = MedicalDataset(breast_dataset, transform=transform)
dataloader = DataLoader(medical_dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# 训练模型
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        
        # 训练判别器
        optimizer_D.zero_grad()
        
        # 生成真实标签和假的标签
        real_labels = torch.ones(imgs.size(0), 1)
        fake_labels = torch.zeros(imgs.size(0), 1)
        
        # 生成噪声
        z = torch.randn(imgs.size(0), latent_dim)
        
        # 生成假图像
        gen_imgs = generator(z, labels)
        
        # 计算判别器的损失
        real_loss = adversarial_loss(discriminator(imgs, labels), real_labels)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        
        # 反向传播和优化
        d_loss.backward()
        optimizer_D.step()
        
        # 训练生成器
        optimizer_G.zero_grad()
        
        # 生成假图像并计算生成器的损失
        gen_imgs = generator(z, labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, labels), real_labels)
        
        # 反向传播和优化
        g_loss.backward()
        optimizer_G.step()
        
        # 打印训练信息
        batches_done = epoch * len(dataloader) + i
        if batches_done % 400 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

        # 保存生成的图像
        if batches_done % 400 == 0:
            save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)
