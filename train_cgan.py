import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
from PIL import Image, ImageDraw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMG_SIZE = 64
CHANNELS = 1
NUM_CLASSES = 3  # 0: Circle, 1: Square, 2: Triangle
LATENT_DIM = 100
EMBEDDING_DIM = 100
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.0002

# 1. Synthetic Shape Dataset Generator
def generate_synthetic_shapes(num_samples=1500):
    images, labels = [], []
    for i in range(num_samples):
        shape_type = i % 3
        img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
        draw = ImageDraw.Draw(img)
        
        if shape_type == 0:    # Circle
            draw.ellipse([12, 12, 52, 52], fill=255)
        elif shape_type == 1:  # Square
            draw.rectangle([14, 14, 50, 50], fill=255)
        elif shape_type == 2:  # Triangle
            draw.polygon([(32, 12), (12, 52), (52, 52)], fill=255)
            
        img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0
        images.append(img_np)
        labels.append(shape_type)
        
    return torch.tensor(np.array(images)).unsqueeze(1), torch.tensor(labels)

# 2. Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBEDDING_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBEDDING_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, CHANNELS * IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        return img.view(img.size(0), CHANNELS, IMG_SIZE, IMG_SIZE)

# 3. Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBEDDING_DIM)
        self.model = nn.Sequential(
            nn.Linear(EMBEDDING_DIM + CHANNELS * IMG_SIZE * IMG_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        return self.model(d_in)

# 4. Training Execution
if __name__ == "__main__":
    os.makedirs("cgan_output", exist_ok=True)
    images, labels = generate_synthetic_shapes()
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels), batch_size=BATCH_SIZE, shuffle=True
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    loss_fn = nn.BCELoss()

    opt_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    print(f"Training CGAN on {device}...")
    for epoch in range(EPOCHS):
        for imgs, lbls in dataloader:
            b_size = imgs.size(0)
            real_t = torch.ones(b_size, 1).to(device)
            fake_t = torch.zeros(b_size, 1).to(device)

            real_imgs, lbls = imgs.to(device), lbls.to(device)

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(b_size, LATENT_DIM).to(device)
            gen_lbls = torch.randint(0, NUM_CLASSES, (b_size,)).to(device)
            gen_imgs = generator(z, gen_lbls)
            g_loss = loss_fn(discriminator(gen_imgs, gen_lbls), real_t)
            g_loss.backward()
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            r_loss = loss_fn(discriminator(real_imgs, lbls), real_t)
            f_loss = loss_fn(discriminator(gen_imgs.detach(), gen_lbls), fake_t)
            d_loss = (r_loss + f_loss) / 2
            d_loss.backward()
            opt_D.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            test_z = torch.randn(6, LATENT_DIM).to(device)
            test_lbls = torch.tensor([0, 0, 1, 1, 2, 2]).to(device)
            save_image(generator(test_z, test_lbls).data, f"cgan_output/epoch_{epoch+1}.png", nrow=2, normalize=True)

    torch.save(generator.state_dict(), "cgan_generator.pth")
    print("Training finished! Generator saved to 'cgan_generator.pth'.")