import os
import json
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

try:
    from rich.progress import track
except ImportError:
    from tqdm import tqdm as track

@dataclass
class Config:
    # Data parameters
    data_path: str = "data/parkinson.csv"
    image_size: Tuple[int, int] = (224, 224)
    sequence_length: int = 30
    num_classes: int = 2
    batch_size: int = 32
    num_workers: int = 4

    # Model parameters
    cnn_backbone: str = "efficientnet_b0"
    rnn_type: str = "gru"
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    fusion_strategy: str = "cross_modal_attention"

    # Training parameters
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    ema_decay: float = 0.9999
    stochastic_depth: float = 0.2
    label_smoothing: float = 0.1

    # Compression parameters
    quantization: bool = False
    pruning: bool = False

    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5

    # Logging parameters
    log_interval: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ParkinsonDataset(Dataset):
    def __init__(self, config: Config, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.data = self._load_data()
        self.transform = self._get_transform()

    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.config.data_path):
            data = pd.read_csv(self.config.data_path)
        else:
            data = self._generate_synthetic_data()

        # Preprocessing
        data = self._preprocess_data(data)
        return data

    def _generate_synthetic_data(self) -> pd.DataFrame:
        num_samples = 1000
        data = {
            "image_path": [f"image_{i}.jpg" for i in range(num_samples)],
            "sequence": [np.random.rand(self.config.sequence_length, 10).tolist() for _ in range(num_samples)],
            "label": [random.randint(0, self.config.num_classes - 1) for _ in range(num_samples)]
        }
        return pd.DataFrame(data)

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Normalize sequences
        sequences = np.array(data["sequence"].tolist())
        sequences = (sequences - np.mean(sequences, axis=1, keepdims=True)) / np.std(sequences, axis=1, keepdims=True)
        data["sequence"] = sequences.tolist()

        # Class weighting
        class_counts = data["label"].value_counts()
        self.class_weights = torch.tensor([1.0 / class_counts[i] for i in range(self.config.num_classes)], dtype=torch.float32)

        return data

    def _get_transform(self) -> transforms.Compose:
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data.iloc[idx]
        image_path = item["image_path"]
        sequence = torch.tensor(item["sequence"], dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)

        # Load image
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(np.random.randint(0, 255, (self.config.image_size[0], self.config.image_size[1], 3), dtype=np.uint8))

        image = self.transform(image)

        return image, sequence, label

class CrossModalAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, cnn_features: torch.Tensor, rnn_features: torch.Tensor) -> torch.Tensor:
        # cnn_features: (batch_size, cnn_features_dim)
        # rnn_features: (batch_size, seq_len, rnn_features_dim)

        # Project features to same dimension
        cnn_features = cnn_features.unsqueeze(1)  # (batch_size, 1, cnn_features_dim)
        rnn_features = rnn_features.permute(0, 2, 1)  # (batch_size, rnn_features_dim, seq_len)

        # Compute attention scores
        query = self.query(cnn_features)  # (batch_size, 1, hidden_size)
        key = self.key(rnn_features)  # (batch_size, hidden_size, seq_len)
        attention_scores = torch.bmm(query, key)  # (batch_size, 1, seq_len)
        attention_weights = self.softmax(attention_scores)  # (batch_size, 1, seq_len)

        # Apply attention weights to rnn_features
        value = self.value(rnn_features)  # (batch_size, hidden_size, seq_len)
        attended_features = torch.bmm(attention_weights, value.permute(0, 2, 1))  # (batch_size, 1, hidden_size)

        # Concatenate with cnn_features
        fused_features = torch.cat([cnn_features, attended_features], dim=-1)  # (batch_size, 1, 2*hidden_size)

        return fused_features.squeeze(1)

class NeuroForgeModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # CNN backbone
        self.cnn_backbone = efficientnet_b0(pretrained=True)
        num_features = self.cnn_backbone.classifier[1].in_features
        self.cnn_backbone.classifier = nn.Identity()

        # RNN
        if config.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=10, hidden_size=config.hidden_size, num_layers=config.num_layers,
                             dropout=config.dropout if config.num_layers > 1 else 0, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=10, hidden_size=config.hidden_size, num_layers=config.num_layers,
                              dropout=config.dropout if config.num_layers > 1 else 0, batch_first=True)

        # LayerNorm for RNN
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Fusion
        if config.fusion_strategy == "cross_modal_attention":
            self.fusion = CrossModalAttention(config)
        else:
            self.fusion = nn.Linear(2 * config.hidden_size, config.hidden_size)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )

        # Stochastic depth
        self.stochastic_depth = config.stochastic_depth

    def forward(self, images: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        # CNN features
        cnn_features = self.cnn_backbone(images)  # (batch_size, num_features)

        # RNN features
        rnn_output, _ = self.rnn(sequences)  # (batch_size, seq_len, hidden_size)
        rnn_features = rnn_output[:, -1, :]  # Take the last time step
        rnn_features = self.layer_norm(rnn_features)  # (batch_size, hidden_size)

        # Fusion
        if self.config.fusion_strategy == "cross_modal_attention":
            fused_features = self.fusion(cnn_features, rnn_output)
        else:
            fused_features = torch.cat([cnn_features, rnn_features], dim=-1)
            fused_features = self.fusion(fused_features)

        # Stochastic depth
        if self.training and self.stochastic_depth > 0:
            mask = torch.bernoulli(torch.full_like(fused_features, 1 - self.stochastic_depth))
            fused_features = fused_features * mask

        # Classifier
        logits = self.classifier(fused_features)

        return logits

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = NeuroForgeModel(config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = self._get_scheduler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.ema = self._get_ema()
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    def _get_scheduler(self) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            else:
                return max(0.0, float(self.config.epochs - current_step) / float(max(1, self.config.epochs - self.config.warmup_steps)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _get_ema(self) -> Optional[torch.optim.swa_utils.AveragedModel]:
        if self.config.ema_decay > 0:
            return torch.optim.swa_utils.AveragedModel(self.model, device=self.device)
        return None

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, sequences, labels in track(dataloader, description="Training"):
            images = images.to(self.device)
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images, sequences)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update_parameters(self.model)

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def _validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, sequences, labels in track(dataloader, description="Validation"):
                images = images.to(self.device)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images, sequences)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_dataloader)
            val_loss, val_accuracy = self._validate(val_dataloader)

            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= 5:
                    print("Early stopping triggered")
                    break

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["val_loss"]
        print(f"Checkpoint loaded from {checkpoint_path}")

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        val_loss, val_accuracy = self._validate(dataloader)
        print(f"Evaluation Results:")
        print(f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        return val_loss, val_accuracy

    def predict(self, image: torch.Tensor, sequence: torch.Tensor) -> int:
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            sequence = sequence.unsqueeze(0).to(self.device)
            logits = self.model(image, sequence)
            _, predicted = torch.max(logits.data, 1)
            return predicted.item()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NeuroForge Parkinson Detection")
    parser.add_argument("--data_path", type=str, default="data/parkinson.csv", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()

    config = Config(
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )

    # Create datasets
    dataset = ParkinsonDataset(config)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Create trainer
    trainer = Trainer(config)

    # Load checkpoint if specified
    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.load_checkpoint)

    # Train the model
    trainer.train(train_dataloader, val_dataloader)

    # Evaluate the model
    trainer.evaluate(val_dataloader)

if __name__ == "__main__":
    main()