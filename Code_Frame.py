import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, img_shape):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z, c):
        # z为随机噪声，c为条件信息
        input = torch.cat((z, c), -1)
        img = self.model(input)
        img = img.view(img.size(0), *img_shape)
        return img

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, img_shape, condition_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(img_shape) + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, c):
        input = torch.cat((img.view(img.size(0), -1), c), -1)
        validity = self.model(input)
        return validity

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# 定义训练函数
def train(generator, discriminator, dataloader, num_epochs, latent_dim, condition_dim, img_shape):
    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]
            valid = torch.ones(batch_size, 1).cuda()
            fake = torch.zeros(batch_size, 1).cuda()

            real_imgs = imgs.cuda()
            labels = labels.cuda()

            # 训练生成器
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim).cuda()
            gen_imgs = generator(z, labels)
            validity = discriminator(gen_imgs, labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()

            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)

            fake_pred = discriminator(gen_imgs.detach(), labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

# 定义生成函数
def generate(generator, num_samples, latent_dim, condition_dim, img_shape):
    z = torch.randn(num_samples, latent_dim).cuda()
    labels = torch.randint(0, 2, (num_samples, 1)).cuda()  # 假设有两种癌变类型
    gen_imgs = generator(z, labels)
    return gen_imgs

# 参数设置
latent_dim = 100
condition_dim = 1
img_shape = (3, 64, 64)  # 假设图像大小为64x64，3通道
num_epochs = 200
batch_size = 64

# 加载数据集
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
dataset = datasets.ImageFolder(root="data_path", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器，并将它们移动到 GPU 上
generator = Generator(latent_dim, condition_dim, img_shape).cuda()
discriminator = Discriminator(img_shape, condition_dim).cuda()

# 训练模型
train(generator, discriminator, dataloader, num_epochs, latent_dim, condition_dim, img_shape)

# 生成新图像
num_samples = 10
generated_images = generate(generator, num_samples, latent_dim, condition_dim, img_shape)
