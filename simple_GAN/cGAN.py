import torch
import numpy as np
import tqdm
import torch.nn as nn
import pandas as pd
from pytorch_lightning import LightningModule ,Trainer
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split

class Generator(nn.Module):
    def __init__(self, latent_dim, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            
            nn.Linear(2048, N),
            nn.Softmax()
        )
    
    def forward(self, y):
        return self.net(y)

    
class Discriminator(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.float()
    
    def forward(self, y):
        return self.net(y)
    
    
class GAN(LightningModule):
    def __init__(self, N, latent_dim, lr, device) :
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.latent_dim = latent_dim
        self.lr =lr
        self.G = Generator(self.hparams.latent_dim , N).to(device)
        self.D = Discriminator(N).to(device)
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        
    def forward(self, z):
        return self.G(z)
    
    def adversarial_loss(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y)
    
    def training_step(self, batch):
        true_sample =batch
        batch_size = 32
        
        # Generator Noise
        z = torch.randn(batch_size, self.hparams.latent_dim, 
                        requires_grad=True, dtype=torch.float32).to(device)
        optimizer_g, optimizer_d = self.optimizers()
        
        # Train Generator
        self.toggle_optimizer(optimizer_g)
        fake_sample = self(z)
        y_pred = self.D(fake_sample).clone().detach().requires_grad_(True)
        g_loss = self.adversarial_loss(y_pred, torch.ones_like(y_pred))
        self.log('g_loss', g_loss)
        self.backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        
        #Train Discriminator
        self.toggle_optimizer(optimizer_d)
        fake_sample = self(z)
        y_fake_pred = self.D(fake_sample).clone().detach().requires_grad_(True).float()
        y_true_pred = self.D(true_sample).clone().detach().requires_grad_(True).float()
        
        real_loss = self.adversarial_loss(y_true_pred, torch.ones_like(y_true_pred))
        fake_loss = self.adversarial_loss(y_fake_pred, torch.zeros_like(y_fake_pred))
        d_loss = (real_loss + fake_loss) / 2
        self.log('d_loss', d_loss)
        self.backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.G.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        


P_train = pd.read_csv('train_abundance.csv')

N = P_train.shape[1]

P_train, P_val = random_split(torch.tensor(P_train.values, dtype=torch.float32), 
                            [0.8, 0.2])
batch_size, lr, latent_dim, epochs = 32, 0.0001, 20, 100

train_loader = DataLoader(P_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(P_val, batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# initialize model
model = GAN(N, latent_dim, lr, device=device).to(device)
grad_clip_val = 1.0 

trainer = Trainer(max_epochs=epochs, enable_progress_bar=20, log_every_n_steps=1, accelerator='gpu')
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint('GANmodel.ckpt')

    