import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from model import JSCC_Classifier
from data_load import load_cifar10_data, load_kodak_dataset

def evaluate_model(model, test_loader, snr_list, channel_type, device):
    """
    评估一个给定的模型在一系列SNR下的分类准确率。
    """
    model.eval() 
    accuracies_over_snr = {} 

    with torch.no_grad(): 
        for snr_db in tqdm(snr_list, desc=f"Evaluating on {channel_type} channel for different SNRs"):
            
            correct = 0
            total = 0
            for data, target in test_loader: 
                data, target = data.to(device), target.to(device)
                output = model(data, snr_db=snr_db, channel_type=channel_type)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            accuracies_over_snr[str(snr_db)] = accuracy 
            
    return accuracies_over_snr

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained JSCC_Classifier model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'kodak'], help="Dataset the model was trained on.")
    parser.add_argument('--channel_types', nargs='+', default=['awgn'], help="List of channel types to evaluate. We focus on 'awgn'.")
    parser.add_argument('--snr_min', type=int, default=-14, help="Minimum SNR in dB for evaluation range.")
    parser.add_argument('--snr_max', type=int, default=12, help="Maximum SNR in dB for evaluation range.")
    parser.add_argument('--snr_step', type=int, default=2, help="Step size for SNR in dB.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save evaluation results.")
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

    test_loader, H, W, C = (None, 0, 0, 0)
    if args.dataset == 'cifar10':
        _, test_loader = load_cifar10_data(args.batch_size)
        H, W, C = 32, 32, 3
        print("CIFAR-10 test data loaded.")
    elif args.dataset == 'kodak':
        test_loader = load_kodak_dataset(path='./kodak_dataset', batch_size=1)
        sample_data, _ = next(iter(test_loader)) # Get a sample to determine dimensions
        _, C, H, W = sample_data.shape
        print("Kodak test data loaded.")
    
    try:
        model_basename = os.path.basename(args.model_path)
        ratio_str = model_basename.split('_kn')[1].split('.pth')[0]
        compression_ratio = float(ratio_str)
        
        latent_H, latent_W = H // 4, W // 4
        latent_channels_k = max(1, round(compression_ratio * (C * H * W) / (latent_H * latent_W)))
        print(f"Inferred k/n ratio: {compression_ratio}, using model parameter k={latent_channels_k}")
        
    except (IndexError, ValueError) as e:
        print(f"错误：无法从模型文件名 '{args.model_path}' 中解析出k/n比率。请确保文件名格式正确。")
        exit()
        
    model = JSCC_Classifier(k=latent_channels_k).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded successfully from {args.model_path}")

    snr_list = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    all_results = {}
    for channel_type in args.channel_types:
        avg_accuracies = evaluate_model(model, test_loader, snr_list, channel_type, device)
        all_results[channel_type] = {'accuracy': avg_accuracies}

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    save_path = os.path.join(args.save_dir, f"evaluation_{model_name}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nEvaluation results for {model_name} saved to {save_path}")


if __name__ == "__main__":
    main()