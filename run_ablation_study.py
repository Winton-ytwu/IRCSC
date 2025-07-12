import torch
import os
import argparse
from tqdm import tqdm
from model import JSCC_Classifier
from data_load import load_cifar10_data
from utils import calculate_importance_weights, awgn_channel

def evaluate_performance(model, test_loader, device, snr_db, M, important_indices, discard_most_important=False):
    model.eval()
    correct = 0
    total = 0
    
    indices_to_keep = []
    if discard_most_important:
        # 丢弃最重要的，保留剩下的 M 个
        indices_to_keep = important_indices[1:M+1] 
    else:
        # 正常IRCSC：保留最重要的 M 个
        indices_to_keep = important_indices[:M]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            clean_features_interleaved = model.encoder(images)
            
            mask = torch.zeros_like(clean_features_interleaved)
            for channel_idx in indices_to_keep:
                mask[:, channel_idx, :, :] = 1
            
            features_to_transmit = clean_features_interleaved * mask
            
            z_to_transmit = torch.complex(features_to_transmit[:, 0::2], features_to_transmit[:, 1::2])
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
    parser = argparse.ArgumentParser(description="Run an ablation study to verify the importance of feature selection.")
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
    
    snr_test_point = 10 
    M_test_value = 7    

    model = JSCC_Classifier(k=args.k_value).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"模型加载自: {args.model_path}")
    
    _, test_loader = load_cifar10_data(batch_size=256)
    
    importance_batch_images, importance_batch_labels = next(iter(test_loader))
    _, important_indices = calculate_importance_weights(model, importance_batch_images.to(device), importance_batch_labels.to(device), device)
    print(f"计算出的特征重要性排名: {important_indices.cpu().numpy()}")

    print(f"\n在SNR={snr_test_point}dB, M={M_test_value} 的条件下进行测试")

    print("\n[实验 A] (智能丢弃最不重要的特征通道)")
    accuracy_A = evaluate_performance(model, test_loader, device, snr_test_point, M=M_test_value, important_indices=important_indices, discard_most_important=False)
    print("\n[实验 B] (故意丢弃最重要的特征)")
    accuracy_B = evaluate_performance(model, test_loader, device, snr_test_point, M=M_test_value, important_indices=important_indices, discard_most_important=True)

    print("\nCalculating baseline performance for comparison...")
    total_channels_C = 2 * args.k_value # 定义 total_channels_C
    
    baseline_accuracy = evaluate_performance(model, test_loader, device, snr_test_point, M=total_channels_C, important_indices=important_indices, discard_most_important=False)

    print("\n--- 实验结论 ---")
    print(f"基准性能 (TD-JSCC, M={total_channels_C}) 在SNR={snr_test_point}dB时为: {baseline_accuracy:.2f}%")
    print(f"实验A (智能丢弃最不重要的特征) 准确率: {accuracy_A:.2f}%")
    print(f"实验B (故意丢弃最重要的特征) 准确率: {accuracy_B:.2f}%")

if __name__ == '__main__':
    main()