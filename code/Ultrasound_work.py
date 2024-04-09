# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
random.seed(42)
#定义超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr=0.0002
batch_size = 128
num_epoch = 50
latent_size = 100
image_size=28*28

# 从CSV文件中读取标签信息并转换为字典
def labels_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    label_dict = dict(zip(df['Image_filename'], df['Classification']))
    return label_dict

# 指定文件夹路径和CSV文件路径
folder_path = "D:\MONAI\CGAN_Image_sup\CGAN_Image\data/all_images"
csv_file = "D:\MONAI\CGAN_Image_sup\CGAN_Image\data\image_labels.csv"

# 将CSV文件中的标签信息转换为字典
label_dict = labels_to_dict(csv_file)

# 图像转换操作
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图片大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 转换为Tensor并进行归一化
])

# 自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, folder_path, label_dict, transform=None):
        self.folder_path = folder_path
        self.label_dict = label_dict
        self.transform = transform
        self.images = os.listdir(folder_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = os.path.join(self.folder_path, self.images[index])
        image = Image.open(img_name).convert('L')
        image = self.transform(image)
        
        label = self.label_dict[self.images[index]]
        label = torch.tensor(label).int()
        
        return image, label

# 创建自定义Dataset实例
train_dataset = CustomDataset(folder_path=folder_path, label_dict=label_dict, transform=transform)
# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %%
#定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=100, label_dim=3, image_dim=28*28):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_embeddings=label_dim, embedding_dim=10)
        self.hidden1 = nn.Sequential(nn.Linear(latent_dim + 10, 256), nn.ReLU())
        self.hidden2 = nn.Sequential(nn.Linear(256, 512), nn.ReLU())
        self.out_layer = nn.Sequential(nn.Linear(512, image_dim), nn.Tanh())

    def forward(self, noise, labels):
        embedded_labels = self.label_embedding(labels)
        x = torch.cat((noise, embedded_labels), dim=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out_layer(x)
        return x

# %%
#定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=28*28, label_dim=3):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_embeddings=label_dim, embedding_dim=10)
        self.hidden1 = nn.Sequential(nn.Linear(input_dim + 10, 256), nn.LeakyReLU(0.2))
        self.hidden2 = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(0.2))
        self.out_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x, labels):
        embedded_labels = self.label_embedding(labels)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, embedded_labels), dim=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out_layer(x)
        return x

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
    for i,(images,labels) in enumerate(train_loader):
        batch_size=images.shape[0]
        real_images=images.view(batch_size,-1).to(device)
        real_labels=labels.to(device)
        #判别器计算真实图像误差
        real_pred=discriminator(real_images,real_labels)
        d_loss_real=loss(real_pred,torch.ones(batch_size,1).to(device))
        #判别器计算虚假图像误差
        noise=torch.randn(batch_size,latent_size).to(device)
        fake_images=generator(noise,labels.to(device))
        fake_pred=discriminator(fake_images,labels.to(device))
        d_loss_fake=loss(fake_pred,torch.zeros(batch_size,1).to(device))

        d_loss=d_loss_fake+d_loss_real
        #更新判别器参数
        discriminator.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        noise=torch.randn(batch_size,latent_size).to(device)
        fake_images=generator(noise,labels.to(device))
        pred=discriminator(fake_images,labels.to(device))
        g_loss=loss(pred,
                    torch.ones(batch_size,1).to(device)) 
        #更新生成器参数
        generator.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i%100==0:
            print('Epoch [{}/{}], Step[{}/{}],d_loss:{:.4f},g_loss:{:.4f}'
                  .format(epoch+1,num_epoch,i+1,len(train_loader),d_loss.item(),g_loss.item()))
            
# %%
import numpy as np
import matplotlib.pyplot as plt
with torch.no_grad():
    noise = torch.randn(10,100).to(device)
    labels = torch.LongTensor(np.arange(10)).to(device)
    samples = generator(noise,labels).cpu().data
    figs,axs=plt.subplots(1,10,figsize=(10,1))
    for j in range(10):
        axs[j].imshow(samples[j].reshape(28,28),cmap='gray')
        axs[j].axis('off')
    plt.show()