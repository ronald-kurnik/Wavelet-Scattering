import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# import torchvision
from torchvision import transforms
# import kymatio
from kymatio.torch import Scattering2D
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
# import os
from pathlib import Path
import pandas as pd
from PIL import Image

# For reproducibility
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

# ------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\Ron\Documents\2025\NN\WaveletScattering\MatlabData")
TRAIN_CSV = DATA_ROOT / "digitTrain.csv"
TEST_CSV  = DATA_ROOT / "digitTest.csv"

IMG_SIZE   = (28, 28)          # all MATLAB images are 28×28 grayscale
BATCH_SIZE = 128
# NUM_WORKERS = 0

# ------------------------------------------------------------
# 2. Custom Dataset (CSV + folder structure)
# ------------------------------------------------------------
class MatlabDigitDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)               # columns: image,digit,angle
        self.root = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]                       # e.g. "image2119.png"
        digit    = int(row["digit"])
        # angle    = float(row["angle"])

        # Images are stored in digit sub-folders → <root>/<digit>/<image>
        img_path = self.root / str(digit) / img_name

        # Load as grayscale (L mode) → (H,W)
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)                 # → torch Tensor (1,28,28)

        return img, digit

# ------------------------------------------------------------
# 3. Transforms
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),               # [0,255] → [0.0,1.0] + adds channel dim
])

train_ds = MatlabDigitDataset(TRAIN_CSV, DATA_ROOT, transform=transform)
test_ds  = MatlabDigitDataset(TEST_CSV,  DATA_ROOT, transform=transform)

print(f"Train samples: {len(train_ds)}")
print(f"Test  samples: {len(test_ds)}")

# Display 20 random images
plt.figure(figsize=(10, 8))
for i in range(20):
    idx = random.randint(0, len(train_ds) - 1)
    img, _ = train_ds[idx]
    plt.subplot(4, 5, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

# Wavelet Scattering
scat = Scattering2D(J=4, shape=(28, 28), L=8, max_order=2)

# 2. Test on one image
img, _ = train_ds[0]                    # (1, 28, 28)
img = img.unsqueeze(0)                  # (1, 1, 28, 28)

with torch.no_grad():
    s = scat(img)
    print("Output shape:", s.shape)
    print("Number of coefficients:", s.shape[1])
    
    
# Test on one image
img, _ = train_ds[0]                    # (1, 28, 28)
img = img.unsqueeze(0)                  # (1, 1, 28, 28)

with torch.no_grad():
    s = scat(img)
    print("Scattering output shape:", s.shape)

def extract_features_batched(ds, scat, batch_size=128, device='cpu'):
    scat = scat.to(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            s = scat(imgs)                          # (B, 1, 417, 1, 1)
            s_log = torch.log1p(s)
            # Extract the coefficient vector: s[:, 0, :, 0, 0]
            f = s_log[:, 0, :, 0, 0]                # (B, 417)
            feats.append(f.cpu().numpy())
            labels.append(lbls.numpy())

    X = np.concatenate(feats, axis=0)   # (N, 417)
    y = np.concatenate(labels, axis=0)  # (N,)
    return X.T, y                       # (417, N), (N,)

# Extract features
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_features, train_labels = extract_features_batched(
    train_ds, scat, batch_size=128, device=device)

test_features, test_labels = extract_features_batched(
    test_ds, scat, batch_size=128, device=device)

print("Train features shape:", train_features.shape)  # Should be (C, 8000)
print("Train labels shape:", train_labels.shape)      # (8000,)

# PCA Model
class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mu = {}
        self.U = {}
        self.S = {}
        self.labels = np.unique(train_labels)
    
    def fit(self, features, labels):
        if len(features.shape) != 2:
            raise ValueError("Features must be a 2D array (n_features, n_samples)")
        n_samples, n_features = features.shape[1], features.shape[0]  # Correct indexing
        self.n_components = min(self.n_components, n_samples - 1, n_features)  # Adjust n_components
        for c in self.labels:
            idx = labels == c
            feats = features[:, idx]
            mu = np.mean(feats, axis=1)[:, np.newaxis]
            self.mu[c] = mu
            centered = feats - mu
            pca = PCA(n_components=self.n_components)
            pca.fit(centered.T)  # (samples, features)
            self.U[c] = pca.components_.T  # (features, n_comp)
            self.S[c] = pca.explained_variance_

model = PCAModel(30)
model.fit(train_features, train_labels)

# PCA Classifier
def pca_classifier(test_features, model):
    L = test_features.shape[1]
    err_matrix = np.full((len(model.labels), L), np.inf)
    for i, c in enumerate(model.labels):
        mu = model.mu[c]
        u = model.U[c]
        s = test_features - mu
        norm_sqx = np.sum(np.abs(s)**2, axis=0)
        proj_sq = np.sum(np.abs(u.T @ s)**2, axis=0)
        err = np.sqrt(norm_sqx - proj_sq)
        err_matrix[i, :] = err
    label_idx = np.argmin(err_matrix, axis=0)
    pred_labels = model.labels[label_idx]
    return pred_labels

pred_labels = pca_classifier(test_features, model)
accuracy = accuracy_score(test_labels, pred_labels) * 100
print(f'Accuracy: {accuracy}%')

# Confusion matrix
plt.figure()
sns.heatmap(confusion_matrix(test_labels, pred_labels), annot=True, fmt='d', cmap='Blues')
plt.title('Test-Set Confusion Matrix -- Wavelet Scattering')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 5 * 5, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc(x)
        return x

cnn_model = SimpleCNN()
optimizer = optim.SGD(cnn_model.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Train with more epochs and monitoring
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
for epoch in range(30):
    cnn_model.train()
    running_loss = 0.0
    for imgs, lbls in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# Predict
def cnn_predict(model, ds):
    model.eval()
    preds = []
    with torch.no_grad():
        for img, _ in ds:
            img = img.unsqueeze(0)
            out = model(img)
            pred = out.argmax(1).item()
            preds.append(pred)
    return np.array(preds)

cnn_pred_labels = cnn_predict(cnn_model, test_ds)
cnn_accuracy = accuracy_score(test_labels, cnn_pred_labels) * 100
print(f'CNN Accuracy: {cnn_accuracy}%')

# CNN Confusion matrix
plt.figure()
sns.heatmap(confusion_matrix(test_labels, cnn_pred_labels), annot=True, fmt='d', cmap='Blues')
plt.title('Test-Set Confusion Matrix -- CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
