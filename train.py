import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import JSCC_Classifier  
from utils import *
from config import CONFIG
from data_load import load_cifar10_data, load_kodak_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep JSCC model on different datasets.")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'kodak'], 
                        help="Dataset to train on.")
    parser.add_argument('--compression_ratio', type=float, required=True, 
                        help="Bandwidth compression ratio (k/n). This is a required argument.")
    parser.add_argument('--snr_db', type=float, default=CONFIG['snr_db'], 
                        help="Signal-to-Noise Ratio in dB for training.")
    parser.add_argument('--channel_type', type=str, choices=['awgn', 'rayleigh'], default=CONFIG['channel_type'], 
                        help="Type of channel for training.")
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], 
                        help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'], 
                        help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'], 
                        help="Batch size for training.")
    # parser.add_argument('--norm_mode', type=str, default='dynamic', choices=['dynamic', 'paper'],
    #                     help="Normalization mode ('dynamic' or 'paper').")
    return parser.parse_args()

def train_model(epochs, batch_size, learning_rate, snr_db, channel_type, compression_ratio, dataset):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU) for training.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")

    if dataset == 'cifar10':
        train_loader, test_loader = load_cifar10_data(batch_size)
    elif dataset == 'kodak':
        train_loader, test_loader = load_kodak_dataset(batch_size)

    sample_data, _ = next(iter(train_loader))
    _, C, H, W = sample_data.shape
    n_dim = C * H * W
    latent_H, latent_W = H // 4, W // 4
    latent_channels_k = max(1, round(compression_ratio * n_dim / (latent_H * latent_W)))
    print(f"\n--- Training on {dataset.upper()} with: SNR={snr_db}dB, Channel={channel_type}, Target k/n={compression_ratio:.4f} (Model latent channels k={latent_channels_k}) ---")
    
    model = JSCC_Classifier(k=latent_channels_k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accuracies = [], [], []
    best_loss = float('inf')
    patience_counter = 0
    patience = 50 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device) 
            output = model(data, snr_db=snr_db, channel_type=channel_type) # output 是 logits
            loss = criterion(output, target) 
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证模型 
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, snr_db=snr_db, channel_type=channel_type)
                total_val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1) # 获取概率最高的类别索引
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        accuracy = 100 * correct / total # 计算准确率百分比
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break
         
    return model, train_losses, val_losses, val_accuracies