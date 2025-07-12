import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

from model import JSCC_Classifier
from data_load import load_cifar10_data
from utils import calculate_importance_weights, awgn_channel

def evaluate_performance(model, test_loader, device, snr_db, M=None, important_indices=None):
    """
    一个通用的评估函数，可以评估TD-JSCC (M=None) 或 IRCSC (给定M和indices) 的性能。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            clean_features_interleaved = model.encoder(images)

            # 如果是IRCSC模式，则进行特征选择（掩码）
            if M is not None and important_indices is not None:
                mask = torch.zeros_like(clean_features_interleaved)
                for i in range(M):
                    channel_idx = important_indices[i]
                    mask[:, channel_idx, :, :] = 1
                
                features_to_transmit = clean_features_interleaved * mask
            else: # TD-JSCC模式，发送全部特征
                features_to_transmit = clean_features_interleaved
            
            z_to_transmit = torch.complex(
                features_to_transmit[:, 0::2],
                features_to_transmit[:, 1::2]
            )
            z_noisy = awgn_channel(z_to_transmit, snr_db)
            x_noisy = torch.empty_like(features_to_transmit)
            x_noisy[:, 0::2] = torch.real(z_noisy)
            x_noisy[:, 1::2] = torch.imag(z_noisy)

            logits = model.decoder(x_noisy)
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Run the final evaluation for IRCSC vs TD-JSCC.")
    parser.add_argument('--model_path', type=str, default='./models/td_jscc_cifar10_snr0_awgn_kn0.0833.pth')
    parser.add_argument('--k_value', type=int, default=4)
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

    snr_list = list(range(-14, 12, 2))
    total_channels_C = 2 * args.k_value # 2*k 个实际通道

    print(f"Loading model from {args.model_path}...")
    model = JSCC_Classifier(k=args.k_value).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print("Loading CIFAR-10 test data for evaluation...")
    _, test_loader = load_cifar10_data(batch_size=256)
    
    importance_batch_images, importance_batch_labels = next(iter(test_loader))
    importance_batch_images = importance_batch_images.to(device)
    importance_batch_labels = importance_batch_labels.to(device)
    print("Pre-calculating a representative importance ranking...")
    _, important_indices = calculate_importance_weights(model, importance_batch_images, importance_batch_labels, device)


    final_results = {"snr": snr_list, "td_jscc_acc": [], "ircsc_acc": [], "ircsc_M": []}

    for snr_db in tqdm(snr_list, desc="Overall Progress"):
        # 1. 性能目标 τ
        print(f"\nEvaluating TD-JSCC baseline at SNR={snr_db}dB...")
        baseline_accuracy = evaluate_performance(model, test_loader, device, snr_db)
        final_results["td_jscc_acc"].append(baseline_accuracy)
        print(f"  > TD-JSCC Accuracy (τ): {baseline_accuracy:.2f}%")

        # 2. 搜索IRCSC的最小M值
        print(f"Searching for optimal M for IRCSC at SNR={snr_db}dB...")
        best_m_for_snr = total_channels_C
        ircsc_final_accuracy = baseline_accuracy

        # 从M=1开始，线性搜索能达到目标性能的最小M
        for M_candidate in range(1, total_channels_C + 1):
            current_accuracy = evaluate_performance(model, test_loader, device, snr_db, M=M_candidate, important_indices=important_indices)
            print(f"  - Testing with M={M_candidate}, Accuracy={current_accuracy:.2f}%")
            
            # 如果当前准确率达到了基准的98%，就认为达标
            if current_accuracy >= baseline_accuracy * 0.98:
                best_m_for_snr = M_candidate
                ircsc_final_accuracy = current_accuracy
                print(f"  > Found optimal M={best_m_for_snr} for this SNR!")
                break # 找到了就停止搜索

        final_results["ircsc_acc"].append(ircsc_final_accuracy)
        final_results["ircsc_M"].append(best_m_for_snr)

    save_path = './results/final_evaluation_results.json'
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\n--- FINAL EVALUATION COMPLETE! Results saved to {save_path} ---")
    print("Final Results:", json.dumps(final_results, indent=4))

if __name__ == '__main__':
    main()