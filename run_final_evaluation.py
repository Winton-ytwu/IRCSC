import torch
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

from model import JSCC_Classifier
from data_load import load_cifar10_data
from utils import calculate_importance_weights, awgn_channel

def evaluate_performance(model, test_loader, device, snr_db, M=None, important_indices=None, selection_mode="top"):
    """
    一个通用的评估函数，可以评估TD-JSCC, IRCSC, WO-FS, WO-IA的性能。
    selection_mode: "top" (选择最重要的) 或 "random" (随机选择).
    """
    model.eval()
    correct = 0
    total = 0

    # 获取总通道数，用于随机选择
    total_channels_C = model.k * 2

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            clean_features_interleaved = model.encoder(images)

            # 如果M不为None，则进行特征选择
            if M is not None:
                mask = torch.zeros_like(clean_features_interleaved)

                indices_to_keep = []
                if selection_mode == "random":
                    # WO-IA模式：从C个通道中随机选择M个
                    shuffled_indices = torch.randperm(total_channels_C)
                    indices_to_keep = shuffled_indices[:M]
                else: # 默认 "top" 模式 (用于IRCSC和WO-FS)
                    # IRCSC/WO-FS模式：选择最重要的M个
                    indices_to_keep = important_indices[:M]

                for channel_idx in indices_to_keep:
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
    parser = argparse.ArgumentParser(description="Run the final evaluation for IRCSC vs TD-JSCC and others.")
    parser.add_argument('--model_path', type=str, default='./models/td_jscc_cifar10_snr0_awgn_kn0.0833.pth')
    parser.add_argument('--k_value', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snr_list = list(range(-14, 13, 2))
    total_channels_C = 2 * args.k_value

    # 为 WO-FS 定义一个固定的M值 
    M_fixed_wo_fs = int(total_channels_C * (3/8))
    print(f"--- WO-FS will use a fixed M = {M_fixed_wo_fs} ---")

    print(f"Loading model from {args.model_path}...")
    model = JSCC_Classifier(k=args.k_value).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print("Loading CIFAR-10 test data for evaluation...")
    _, test_loader = load_cifar10_data(batch_size=256)

    print("Pre-calculating a representative importance ranking...")
    importance_batch_images, importance_batch_labels = next(iter(test_loader))
    _, important_indices = calculate_importance_weights(model, importance_batch_images.to(device), importance_batch_labels.to(device), device)

    final_results = {
        "snr": snr_list, 
        "td_jscc_acc": [], 
        "ircsc_acc": [], "ircsc_M": [],
        "wo_fs_acc": [],   
        "wo_ia_acc": []    
    }

    for snr_db in tqdm(snr_list, desc="Overall Progress"):
        print(f"\nEvaluating TD-JSCC baseline at SNR={snr_db}dB...")
        baseline_accuracy = evaluate_performance(model, test_loader, device, snr_db)
        final_results["td_jscc_acc"].append(baseline_accuracy)

        print(f"Searching for optimal M for IRCSC at SNR={snr_db}dB...")
        best_m_for_snr = total_channels_C
        ircsc_final_accuracy = baseline_accuracy
        for M_candidate in range(1, total_channels_C + 1):
            current_accuracy = evaluate_performance(model, test_loader, device, snr_db, M=M_candidate, important_indices=important_indices)
            if current_accuracy >= baseline_accuracy * 0.98:
                best_m_for_snr = M_candidate
                ircsc_final_accuracy = current_accuracy
                break
        final_results["ircsc_acc"].append(ircsc_final_accuracy)
        final_results["ircsc_M"].append(best_m_for_snr)

        print(f"Evaluating WO-FS (Fixed M={M_fixed_wo_fs}) at SNR={snr_db}dB...")
        accuracy_wo_fs = evaluate_performance(model, test_loader, device, snr_db, M=M_fixed_wo_fs, important_indices=important_indices)
        final_results["wo_fs_acc"].append(accuracy_wo_fs)

        print(f"Evaluating WO-IA (Random M={best_m_for_snr}) at SNR={snr_db}dB...")
        accuracy_wo_ia = evaluate_performance(model, test_loader, device, snr_db, M=best_m_for_snr, selection_mode="random")
        final_results["wo_ia_acc"].append(accuracy_wo_ia)

    save_path = f'./results/final_evaluation_results_all_k{args.k_value}.json'
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\n--- FINAL EVALUATION COMPLETE! Results saved to {save_path} ---")

if __name__ == '__main__':
    main()
