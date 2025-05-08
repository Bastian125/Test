#!/usr/bin/env -S submit -M 4000 -m 7500 -f python -u

import numpy as np
from natsort import natsorted
import xarray as xr
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as cm
from codecarbon import EmissionsTracker
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# Emission Tracker
tracker = EmissionsTracker(
    project_name="solar_autoencoder_v1",
    tracking_mode="process",
    log_level="error"
)

# Constants
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 800
PATIENCE = 20
CHECKPOINT_PATH = "best_model.pt"

# Optional: skip initial plotting unless set True
beginning_plot = False
if beginning_plot:
    ds = xr.open_dataset("/net/data/erum_data/all128.nc")
    print(f"{ds.channel=}")
    selected_channel = ds['DN'].sel(channel='171A')
    print(f"{selected_channel=}")
    img = selected_channel.isel(time=0)
    cmap = plt.get_cmap('sdoaia171')
    img.plot(cmap=cmap)
    plt.savefig("single.png")
    plt.close()

    keys = natsorted(ds['channel'].data)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, key in zip(axes.ravel(), keys):
        data = ds['DN'].sel(channel=key).isel(time=0)
        cmap = plt.get_cmap(f'sdoaia{key[:-1]}')
        im = data.plot(cmap=cmap, ax=ax, add_colorbar=False)
        ax.set_title(key)
        ax.axis('off')
    plt.savefig("multiple.png")
    plt.close()

# Model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.dec1 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec3 = self.conv_block(64 + 32, 32)
        self.out = nn.Conv2d(32, out_channels, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        dec1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(enc4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(dec2), enc1], dim=1))
        return self.out(dec3)
        
class UNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=8):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)             # [B, 32, 128, 128]
        e2 = self.enc2(self.pool1(e1))# [B, 64, 64, 64]
        e3 = self.enc3(self.pool2(e2))# [B, 128, 32, 32]

        b = self.bottleneck(self.pool3(e3))  # [B, 256, 16, 16]

        d3 = self.up3(b)             # [B, 128, 32, 32]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)            # [B, 64, 64, 64]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)            # [B, 32, 128, 128]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)        # [B, 8, 128, 128]


class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.pool3 = nn.MaxPool2d(2)  # 32 -> 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 16 -> 32
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # 32 -> 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # 64 -> 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )

        self.output_layer = nn.Conv2d(32, 8, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # [B, 32, 128, 128]
        x2 = self.enc2(self.pool1(x1))  # [B, 64, 64, 64]
        x3 = self.enc3(self.pool2(x2))  # [B, 128, 32, 32]
        x_bottleneck = self.bottleneck(self.pool3(x3))  # [B, 256, 16, 16]

        # Decoder + skip connections
        x = self.up3(x_bottleneck)  # [B, 128, 32, 32]
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)  # [B, 64, 64, 64]
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)  # [B, 32, 128, 128]
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.output_layer(x)  # [B, 8, 128, 128]

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 8, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Dataset
class SolarNetCDFDataset(Dataset):
    def __init__(self, path):
        self.ds = xr.open_dataset(path)
        self.input_channel = '94A'
        self.target_channels = [ch for ch in self.ds.channel.data if ch != self.input_channel]
        self.X = self.ds['DN'].sel(channel=self.input_channel)
        self.y = self.ds['DN'].sel(channel=self.target_channels)

    def __len__(self):
        return self.X.sizes['time']

    def __getitem__(self, idx):
        x = self.X.isel(time=idx).values.astype(np.float32)
        y = self.y.isel(time=idx).values.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        return torch.tensor(x), torch.tensor(y)

# Load dataset
dataset = SolarNetCDFDataset("/net/data/erum_data/all128.nc")
total_len = len(dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len],
                                            generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early stopping and loss tracking
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []

tracker.start()
# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            val_preds = model(x_val)
            val_loss = criterion(val_preds, y_val)
            total_val_loss += val_loss.item()

    print(f"Epoch {epoch+1}: Train Loss = {total_train_loss:.4f}, Val Loss = {total_val_loss:.4f}")

    train_losses.append(total_train_loss)
    val_losses.append(total_val_loss)

    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Validation loss improved — model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered.")
        break
tracker.stop()


# Save training history plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_history.png")
plt.close()

# Save loss history to CSV
loss_df = pd.DataFrame({
    "epoch": list(range(1, len(train_losses) + 1)),
    "train_loss": train_losses,
    "val_loss": val_losses
})
loss_df.to_csv("loss_history.csv", index=False)

# Final evaluation on test set
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()
with torch.no_grad():
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    preds = model(x_batch)

# Visualization
i = 0  # sample index
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
axs[0, 0].imshow(x_batch[i, 0].detach().cpu().numpy(), cmap='gray')
axs[0, 0].set_title("Input: 94A")
axs[0, 0].axis('off')

channel_names = [ch for ch in dataset.ds.channel.data if ch != '94A']
for j in range(8):
    ax = axs[(j + 1) // 3, (j + 1) % 3]
    ax.imshow(preds[i, j].detach().cpu().numpy(), cmap='gray')
    ax.set_title(f"Pred: {channel_names[j]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("test_prediction.png")

fig, axs = plt.subplots(2, 8, figsize=(16, 4))
for j in range(8):
    axs[0, j].imshow(y_batch[i, j].detach().cpu().numpy(), cmap='gray')
    axs[0, j].set_title(f"Original: {channel_names[j]}")
    axs[0, j].axis('off')

    axs[1, j].imshow(preds[i, j].detach().cpu().numpy(), cmap='gray')
    axs[1, j].set_title(f"Pred: {channel_names[j]}")
    axs[1, j].axis('off')

plt.tight_layout()
plt.savefig("test_comparison.png")
