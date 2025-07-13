import json
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot the final comparison results.")
    parser.add_argument('--result_file', type=str, required=True,
                        help="Path to the final evaluation result JSON file (e.g., ..._all_k4.json).")
    parser.add_argument('--k_value', type=int, required=True, 
                        help="The 'k' value used in the model.")
    args = parser.parse_args()

    try:
        with open(args.result_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到结果文件 {args.result_file}")
        return

    snr_list = results['snr']
    td_jscc_acc = results['td_jscc_acc']
    ircsc_acc = results['ircsc_acc']
    ircsc_M = results['ircsc_M']
    wo_fs_acc = results['wo_fs_acc']
    wo_ia_acc = results['wo_ia_acc']

    total_channels_C = 2 * args.k_value
    td_jscc_M = [total_channels_C] * len(snr_list)

    M_fixed_wo_fs = int(total_channels_C * 0.75)
    # 创建一个长度和snr_list相同，但值全部为M_fixed_wo_fs的列表
    wo_fs_M = [M_fixed_wo_fs] * len(snr_list)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
    fig.suptitle(f'Full Comparison (k={args.k_value}) on CIFAR-10 (AWGN)', fontsize=16)
    ax1.plot(snr_list, td_jscc_acc, marker='*', linestyle='--', label='TD-JSCC (Baseline)', color='red')
    ax1.plot(snr_list, ircsc_acc, marker='d', linestyle='-', label='IRCSC (Ours)', color='black')
    ax1.plot(snr_list, wo_fs_acc, marker='x', linestyle=':', label='WO-FS (Fixed M)', color='green')
    ax1.plot(snr_list, wo_ia_acc, marker='o', linestyle='-.', label='WO-IA (Random M)', color='orange')

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(snr_list)
    ax1.set_ylim(0, 100)

    ax2.plot(snr_list, td_jscc_M, marker='*', linestyle='--', label=f'TD-JSCC (M=C={total_channels_C})', color='red')
    ax2.plot(snr_list, ircsc_M, marker='d', linestyle='-', label='IRCSC (Adaptive M)', color='black')
    ax2.plot(snr_list, wo_fs_M, marker='x', linestyle=':', label=f'WO-FS (Fixed M={M_fixed_wo_fs})', color='green')
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

    print(f"\n--- 对比图已成功生成！请查看: {save_path} ---")


if __name__ == '__main__':
    main()
