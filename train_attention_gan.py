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
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0002

# --- Self-Attention Layer for GANs ---
class SelfAttention(nn.Module):
    """Self-attention layer designed to help the model focus on global spatial relationships."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Project to query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        
        # Attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Apply attention weights back
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Scale and add residual connection
        return self.gamma * out + x

# --- 1. Synthetic Shape Dataset Generator ---
def generate_synthetic_shapes(num_samples=1200):
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

# --- 2. Attention-Enhanced Generator ---
class AttentionGenerator(nn.Module):
    def __init__(self):
        super(AttentionGenerator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBEDDING_DIM)
        
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBEDDING_DIM, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 16 * 16),
            nn.ReLU(True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            SelfAttention(128), # Integrated Self-Attention Layer
            nn.ConvTranspose2d(128, CHANNELS, 4, stride=2, padding=1), # 64x64
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        out = self.fc(gen_input)
        out = out.view(out.size(0), 256, 16, 16)
        img = self.conv_blocks(out)
        return img

# --- 3. Discriminator Network ---
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

# --- 4. Training Loop ---
if __name__ == "__main__":
    os.makedirs("attention_gan_output", exist_ok=True)
    images, labels = generate_synthetic_shapes()
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels), batch_size=BATCH_SIZE, shuffle=True
    )

    generator = AttentionGenerator().to(device)
    discriminator = Discriminator().to(device)
    loss_fn = nn.BCELoss()

    opt_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    print(f"Training Attention-Enhanced GAN on {device}...")
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


        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            test_z = torch.randn(6, LATENT_DIM).to(device)
            test_lbls = torch.tensor([0, 0, 1, 1, 2, 2]).to(device)
            save_image(generator(test_z, test_lbls).data, f"attention_gan_output/epoch_{epoch+1}.png", nrow=2, normalize=True)

    torch.save(generator.state_dict(), "attention_gan_generator.pth")
    print("Training finished! Attention-enhanced generator saved to 'attention_gan_generator.pth'.")