import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
sample_dir = 'samples'
import matplotlib.pyplot as plt
import numpy
#模型参数
num_epoches = 100
Batch_size = 1
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#数据预处理
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

        
class Conv(nn.Module):
    """Unet卷积层，两层 (nn.conv2d—nn.BatchNorm2d—nn.LeakReLU)组成
        1. 数据形状
        ->>输入: (batch, in_channel, image_h, image_w)
        ->>输出: (batch, out_channel, image_h, image_w)
        2. 作用： 是将输入数据的通道个数由in_channel变为out_channel
    """
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.layer(x)

class DownSample(nn.Module):
    """下采样层
        1. 可选择：
        ->>model="conv"卷积的方式采样；用卷积将保留更多特征
        ->>model="maxPool"最大池化的方式进行采样。若采用该方法，将不用输入通道数目
        2. 默认使用maxPool的方式进行下采样。
        3. 数据形状：
        ->> 输入: (batch, in_channel, image_h, image_w)
        ->> 输出: (batch, in_channel, image_h/2, image_w/2)
        4. 作用：将图像大小缩小一半"""
    def __init__(self, channel=None, model="maxPool"):
        super(DownSample, self).__init__()
        if model == "conv":
            self.layer=nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(2, 2), stride=(2, 2), bias=False),
                nn.LeakyReLU(inplace=True)
            )
        if model == "maxPool":
            self.layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    """上采样层"""
    def __init__(self, in_channels, out_channels, scale=2):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)  # 使用nn.Upsample进行上采样
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 使用3x3的卷积层来调整通道数

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    """输入(batch,(time_len channel),image_h,image_w)
    返回(batch (time_len channel) image_h image_w)"""
    def __init__(self):
        super(Unet, self).__init__()
# ---------------------------------下采样阶段-------------------------------
        # 下采样阶段压缩图片
        self.d_c0 = Conv(1, 64)
        self.d_s0 = DownSample(64)
        self.d_c1 = Conv(64, 128)
        self.d_s1 = DownSample(128)
        self.d_c2 = Conv(128, 256)
        self.d_s2 = DownSample(256)
        self.d_c3 = Conv(256, 512)
        self.d_s3 = DownSample(512)
        self.d_c4 = Conv(512, 1024)
# -------------------------------------------------------------------------
        #self.middle = Conv(1024, 512)
        self.middle_up = UpSample(1024,512)
# -------------------------------上采样阶段-----------------------------------
        # 上采样阶段将图片还原
        self.u_c4 = Conv(1024, 512)
        self.u_s4 = UpSample(512,256)
        self.u_c3 = Conv(512, 256)
        self.u_s3 = UpSample(256,128)
        self.u_c2 = Conv(256, 128)
        self.u_s2 = UpSample(128,64)
        self.u_c1 = Conv(128, 64)
        self.u_c0 = Conv(64, 1)
        #self.output = nn.Tanh()
# ------------------------------------------------------------------------------
    def forward(self, x):
        d_c0_output = self.d_c0(x)
        #print("d_c0_output shape:", d_c0_output.shape)  # 打印d_c0_output的形状
        d_c1_output = self.d_c1(self.d_s0(d_c0_output))
        #print("d_c1_output shape:", d_c1_output.shape)
        d_c2_output = self.d_c2(self.d_s1(d_c1_output))
        #print("d_c2_output shape:", d_c2_output.shape)
        d_c3_output = self.d_c3(self.d_s2(d_c2_output))
        #print("d_c3_output shape:", d_c3_output.shape)
        d_s4_output = self.d_c4(self.d_s3(d_c3_output))
        #print("d_s4_output shape:", d_s4_output.shape)
        middle_output = self.middle_up(d_s4_output)
        #print("middle_output shape:", middle_output.shape)
        u_s4_output = self.u_s4(self.u_c4(self.cat(middle_output, d_c3_output)))
       # print("u_s4_output shape:", u_s4_output.shape)
        u_s3_output = self.u_s3(self.u_c3(self.cat(u_s4_output, d_c2_output)))
        #print("u_s3_output shape:", u_s3_output.shape)
        u_s2_output = self.u_s2(self.u_c2(self.cat(u_s3_output, d_c1_output)))
        #print("u_s2_output shape:", u_s2_output.shape)
        u_c1_output = self.u_c1(self.cat(u_s2_output, d_c0_output))
        #print("u_c1_output shape:", u_c1_output.shape)
        #print(x.shape)
        output = self.u_c0(u_c1_output)
        #output = self.output(output)
        return output
    def cat(self, x1, x2):
        """在通道维度上组合"""
        return torch.cat([x1, x2], dim=1)

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
    
transform = transforms.Compose([
    transforms.ToTensor(), 
])

MyDataset = PairedImageDataset(root_dir=r'D:\MONAI\CGAN_Image_sup\uct_pair_data_dachaung\uct_pair_data_dachaung',transform=transform)
MyDataloader = torch.utils.data.DataLoader(MyDataset,batch_size=Batch_size,shuffle=True)
generator = Unet()
generator = generator.to(device)
discriminator = PatchGAN().to(device)
g_Optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
d_Optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
lossF = nn.BCELoss()
lossF = lossF.to(device)
for epoch in range(num_epoches):
    for i, (X,Y) in enumerate(MyDataloader):
        #print('start training! step{} epoch{}'.format(i,epoch))
        
        #先训练判别器
        discriminator.zero_grad()
        real_output = discriminator(Y,X)
        fake_image = generator(X)
        fake_output = discriminator(fake_image.detach(),X) #不更新生成器
        d_loss_real = lossF(real_output, torch.ones_like(real_output))
        d_loss_fake = lossF(fake_output, torch.zeros_like(fake_output))
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        d_Optimizer.step()
        
        #再训练生成器
        generator.zero_grad()
        fake_output = discriminator(fake_image,X)
        g_loss = lossF(fake_output, torch.ones_like(fake_output))  # 欺骗判别器
        g_loss.backward()
        g_Optimizer.step()

        images = fake_image.reshape(Y.size(0), 1, 1024, 1024)
        images = fake_image
        save_image(images,os.path.join(sample_dir, 'fake_images-{}.png'.format(i + 1)))
        print(f"Epoch [{epoch}/{num_epoches}], Step [{i}/{len(MyDataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")