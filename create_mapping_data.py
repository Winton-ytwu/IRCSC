# create_mapping_data.py (最终正确版 v4)

import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

from model import JSCC_Classifier
from data_load import load_cifar10_data
from utils import calculate_importance_weights, calculate_stii, calculate_ber_rayleigh, awgn_channel


def find_min_M_for_stii(sorted_weights, C, ber, target_eta):
    low_m, high_m = 1, C
    best_m = C
    while low_m <= high_m:
        mid_m = (low_m + high_m) // 2
        current_stii = calculate_stii(sorted_weights, mid_m, ber)
        if current_stii >= target_eta:
            best_m = mid_m
            high_m = mid_m - 1
        else:
            low_m = mid_m + 1
    return best_m

def main():
    parser = argparse.ArgumentParser(description="Generate data for STII-Accuracy mapping function.")
    parser.add_argument('--model_path', type=str, default='./models/td_jscc_cifar10_snr0_awgn_kn0.0833.pth', help="Path to the trained TD-JSCC model.")
    parser.add_argument('--k_value', type=int, default=4, help="The 'k' value of the trained model.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU) for evaluation.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for evaluation.")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation.")

    target_etas = np.arange(0.5, 1.0, 0.05).tolist()
    snr_range = (-14, 12)

    print(f"Loading model from {args.model_path}...")
    model = JSCC_Classifier(k=args.k_value).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("Loading CIFAR-10 test data...")
    _, test_loader = load_cifar10_data(batch_size=128)

    mapping_data = {}

    for eta in tqdm(target_etas, desc="Processing Target η"):
        total_correct = 0
        total_samples = 0
        
        for images, labels in tqdm(test_loader, desc=f"  Evaluating for η={eta:.2f}", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
            ber = calculate_ber_rayleigh(snr_db)

            sorted_weights, important_indices = calculate_importance_weights(model, images, labels, device)
            
            M = find_min_M_for_stii(sorted_weights, len(sorted_weights), ber, eta)

            with torch.no_grad():
                clean_features_interleaved = model.encoder(images)
                mask = torch.zeros_like(clean_features_interleaved)
                for i in range(M):
                    channel_idx = important_indices[i]
                    mask[:, channel_idx, :, :] = 1
                
                masked_features_interleaved = clean_features_interleaved * mask
                
                z_masked = torch.complex(masked_features_interleaved[:, 0::2], masked_features_interleaved[:, 1::2])
                z_noisy_masked = awgn_channel(z_masked, snr_db)
                x_noisy_masked = torch.empty_like(masked_features_interleaved)
                x_noisy_masked[:, 0::2] = torch.real(z_noisy_masked)
                x_noisy_masked[:, 1::2] = torch.imag(z_noisy_masked)
                
                logits = model.decoder(x_noisy_masked)

                _, predicted = torch.max(logits.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        avg_accuracy = 100 * total_correct / total_samples
        mapping_data[round(eta, 2)] = round(avg_accuracy, 2)
        print(f"  Result: For target η≈{eta:.2f}, Average Accuracy = {avg_accuracy:.2f}%")

    save_path = './results/mapping_data_v4_final.json'
    with open(save_path, 'w') as f:
        json.dump(mapping_data, f, indent=4)
    
    print(f"\n--- FINAL Mapping data generation complete! Results saved to {save_path} ---")

if __name__ == '__main__':
    main()