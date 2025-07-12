import json
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot the final comparison results for IRCSC vs TD-JSCC.")
    parser.add_argument('--result_file', type=str, required=True,
                        help="Path to the final evaluation result JSON file (e.g., ...k4.json or ...k16.json).")
    parser.add_argument('--k_value', type=int, required=True, 
                        help="The 'k' value used in the model to set the correct y-axis limit.")
    args = parser.parse_args()

    try:
        with open(args.result_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到结果文件 {args.result_file}")
        return

    #  提取数据
    snr_list = results['snr']
    td_jscc_acc = results['td_jscc_acc']
    ircsc_acc = results['ircsc_acc']
    ircsc_M = results['ircsc_M']
    
    total_channels_C = 2 * args.k_value
    td_jscc_M = [total_channels_C] * len(snr_list)

    print(f"--- 正在为 k={args.k_value} 的结果生成对比图... ---")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    fig.suptitle(f'IRCSC vs. TD-JSCC (k={args.k_value}, C={total_channels_C}) on CIFAR-10 (AWGN)', fontsize=16)

    ax1.plot(snr_list, td_jscc_acc, marker='*', linestyle='--', label='TD-JSCC Baseline', color='red')
    ax1.plot(snr_list, ircsc_acc, marker='d', linestyle='-', label='IRCSC (Adaptive M)', color='black')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(snr_list)
    ax1.set_ylim(0, 100)

    ax2.plot(snr_list, td_jscc_M, marker='*', linestyle='--', label=f'TD-JSCC (M=C={total_channels_C})', color='red')
    ax2.plot(snr_list, ircsc_M, marker='d', linestyle='-', label='IRCSC (Adaptive M)', color='black')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Number of Transmitted Features (M)')
    ax2.set_title('Transmission Rate Comparison')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(snr_list)
    ax2.set_yticks(np.arange(0, total_channels_C + 2, max(1, total_channels_C // 8)))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_dir = './results'
    base_name = os.path.splitext(os.path.basename(args.result_file))[0]
    save_filename = f"plot_{base_name}.png"
    save_path = os.path.join(save_dir, save_filename)
    
    plt.savefig(save_path)

    print(f"\n--- 对比图已成功生成: {save_path} ---")


if __name__ == '__main__':
    main()