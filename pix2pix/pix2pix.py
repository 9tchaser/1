import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision import transforms

import time
#模型参数
num_epoches = 100
Batch_size = 1
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sample_dir = 'samples'
checkpoint_dir = 'checkpoints'
#创建样本图片路径
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
#创建模型保存路径
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
#权重初始化    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#定义生成器
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Unet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
#定义判别器    
class PatchGAN(nn.Module):
    def __init__(self, in_channels=1):
        super(PatchGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
            
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        output = self.model(img_input)
        # 将输出应用Sigmoid函数，得到概率图
        prob_map = torch.sigmoid(output)
        return prob_map
#定义数据对
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 目录路径，该目录下应有两类子目录，分别存储一对图像。
            transform (callable, optional): 对图像进行预处理的可选变换。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 假设一类图像放在名为"A"的子目录，另一类放在名为"B"的子目录
        self.image_pairs = []
        for idx in os.listdir(os.path.join(root_dir, 'tx8/train')):
            if os.path.isfile(os.path.join(root_dir, 'tx8/train', idx)) and \
                    os.path.isfile(os.path.join(root_dir, 'tx512/train', idx)):
                self.image_pairs.append((os.path.join('tx8/train', idx), os.path.join('tx512/train', idx)))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_path_tx8 = os.path.join(self.root_dir, self.image_pairs[idx][0])
        img_path_tx512 = os.path.join(self.root_dir, self.image_pairs[idx][1])

        image_tx8 = Image.open(img_path_tx8)
        image_tx512 = Image.open(img_path_tx512)

        if self.transform:
            image_tx8 = self.transform(image_tx8)
            image_tx512 = self.transform(image_tx512)
        image_tx512 = image_tx512.to(device)
        image_tx8  = image_tx8.to(device)
        return image_tx8, image_tx512
#转化成张量   
transform = transforms.Compose([
    transforms.ToTensor(), 
])
#设置
MyDataset = PairedImageDataset(root_dir=r'D:\MONAI\CGAN_Image_sup\uct_pair_data_dachaung\uct_pair_data_dachaung',transform=transform)
MyDataloader = DataLoader(MyDataset,batch_size=Batch_size,shuffle=True)
MyDataloader = MyDataloader
generator = Unet().to(device)
discriminator = PatchGAN().to(device)
#初始化权重
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
g_Optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
d_Optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
lossF = nn.BCELoss().to(device)
loss_pixelwise = torch.nn.L1Loss().to(device)
checkpoint_interval = 10
def train_epoch(epoch):
    for i, (X,Y) in enumerate(MyDataloader):
        start_time = time.time()
        X = X.to(device)
        Y = Y.to(device)

        fake_image = generator(X)
        
        # 先训练判别器
        discriminator.zero_grad()
        real_output = discriminator(Y,X)
        fake_output = discriminator(fake_image.detach(),X) # 不更新生成器
        d_loss_real = lossF(real_output, torch.ones_like(real_output))
        d_loss_fake = lossF(fake_output, torch.zeros_like(fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        d_Optimizer.step()
        
        # 再训练生成器
        generator.zero_grad()
        fake_output = discriminator(fake_image,X)
        g_loss = lossF(fake_output, torch.ones_like(fake_output))  # 欺骗判别器
        loss_pix = loss_pixelwise(fake_image, Y)
        g_loss = g_loss + loss_pix
        g_loss.backward()
        g_Optimizer.step()

        images = fake_image.reshape(Y.size(0), 1, 1024, 1024)
        #images = fake_image
        save_image(images,os.path.join(sample_dir, 'fake_images-{}.png'.format(i + 1)))
        end_time = time.time()
        print(f"Epoch [{epoch}/{num_epoches}], Step [{i}/{len(MyDataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Time: {end_time - start_time:.2f} s")
        # 在适当的位置添加释放GPU内存的代码
        if i % 100 == 0:
            torch.cuda.empty_cache()

# 检查是否有最新的 checkpoint
latest_generator_checkpoint = os.path.join(checkpoint_dir, 'latest_generator.pt')
latest_discriminator_checkpoint = os.path.join(checkpoint_dir, 'latest_discriminator.pt')

if os.path.exists(latest_generator_checkpoint) and os.path.exists(latest_discriminator_checkpoint):
    # 加载最新的生成器和判别器状态
    generator.load_state_dict(torch.load(latest_generator_checkpoint))
    discriminator.load_state_dict(torch.load(latest_discriminator_checkpoint))
    print("Latest checkpoints loaded.")

# 从最新的轮数开始继续训练
start_epoch = 0
if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    generator_checkpoints = [f for f in files if f.startswith('generator_epoch_')]
    discriminator_checkpoints = [f for f in files if f.startswith('discriminator_epoch_')]
    if generator_checkpoints and discriminator_checkpoints:
        latest_generator_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in generator_checkpoints])
        latest_discriminator_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in discriminator_checkpoints])
        start_epoch = max(latest_generator_epoch, latest_discriminator_epoch)
        print(f"Resuming training from epoch {start_epoch + 1}.")

# 继续训练
for epoch in range(start_epoch, num_epoches):
    train_epoch(epoch)
    # 在每个 epoch 结束后保存模型状态
    if (epoch + 1) % checkpoint_interval == 0:
        # 保存生成器的状态
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch + 1}.pt'))
        # 保存判别器的状态
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch + 1}.pt'))
        # 保存最新的生成器和判别器状态
        torch.save(generator.state_dict(), latest_generator_checkpoint)
        torch.save(discriminator.state_dict(), latest_discriminator_checkpoint)
        print(f"Checkpoint saved at epoch {epoch + 1}")