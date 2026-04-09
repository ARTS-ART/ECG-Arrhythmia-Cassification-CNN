import os
import yaml
import torch
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from models.models1d import EcgResNet34  # or change to HeartNet
from datasets.dataset_1d import EcgDataset1D
from sklearn.metrics import classification_report
import argparse

# Argument parser for config path
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()
config_path = args.config

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Load YAML config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

set_seed(config["seed"])
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load datasets
train_dataset = EcgDataset1D(config["train_json"], config["class_map"])
val_dataset = EcgDataset1D(config["val_json"], config["class_map"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# Load model
model = EcgResNet34(num_classes=config["num_classes"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# Scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"]
)

# Train loop
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        inputs = batch["image"].to(device)
        labels = batch["class"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % config["log_interval"] == 0:
            print(f"[Epoch {epoch+1}/{config['epochs']}] Batch {i}, Loss: {loss.item():.4f}")

    print(f"📉 Epoch [{epoch+1}/{config['epochs']}], Avg Loss: {total_loss / len(train_loader):.4f}")
    scheduler.step()

    # Evaluation
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            labels = batch["class"].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy())

    print(f"\n📊 Validation Results (Epoch {epoch+1}):")
    print(classification_report(y_true, y_pred, digits=4))

# Save model
os.makedirs(config["save_path"], exist_ok=True)
save_path = os.path.join(config["save_path"], config["save_name"] + ".pth")
torch.save(model.state_dict(), save_path)
print(f"\n✅ Model saved to {save_path}")
