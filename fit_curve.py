# fit_curve.py

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def rational_function(eta, p1, p2, p3, p4, q1, q2):
    """
    定义论文中的有理函数模型，即公式(11)。
    """
    numerator = p1 * eta**3 + p2 * eta**2 + p3 * eta + p4
    denominator = eta**2 + q1 * eta + q2
    return numerator

def main():
    data_path = './results/mapping_data_v4_final.json'
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        return

    eta_values = np.array([float(k) for k in data.keys()])
    accuracy_values = np.array(list(data.values()))
    
    try:
        #使用曲线拟合来找到最优参数
        params, _ = curve_fit(rational_function, eta_values, accuracy_values)
        
        #打印模型参数
        param_names = ['p1', 'p2', 'p3', 'p4', 'q1', 'q2']
        print("\n映射函数 φ(η) 参数如下:")
        for name, value in zip(param_names, params):
            print(f"  {name}: {value:.4f}")

        
        plt.figure(figsize=(10, 7))
        plt.scatter(eta_values, accuracy_values, label='Collected Data Points (Ours)', color='red', zorder=5)
        
        eta_smooth = np.linspace(min(eta_values), max(eta_values), 100)
        accuracy_fit = rational_function(eta_smooth, *params) 
        plt.plot(eta_smooth, accuracy_fit, label='Fitted Function φ(η)', color='blue')
        
        plt.title('STII vs. Accuracy Mapping Function')
        plt.xlabel('STII (η)')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        save_path = './results/mapping_function_plot.png'
        plt.savefig(save_path)
        print(f"\n✓ 拟合结果图已保存至: {save_path}")

    except RuntimeError:
        print("\n错误：曲线拟合失败。可能是因为数据点太少或噪声太大，无法收敛。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == '__main__':
    main()