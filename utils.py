import torch
import torch.nn.functional as F
import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.nn.functional as F
import torch.autograd as autograd



# AWGN Channel
def awgn_channel(z, snr_db):

    power = torch.mean(torch.abs(z) ** 2) # 计算实际平均功率P
    snr_linear = 10 ** (snr_db / 10) # 将dB转换为线性比例, P / σ²
    noise_power = power / snr_linear # σ²
    # torch.randn_like(real)生成一个与信号实部形状相同、符合标准正态分布（功率/方差=1）的“原始”噪声。
    # torch.sqrt(noise_power / 2) 计算出目标实部噪声的“标准差”。（总功率均分给实部和虚部，再开方得到标准差）
    # 两者相乘，将“原始”噪声缩放到期望的正确功率水平。
    noise_real = torch.randn_like(z.real) * torch.sqrt(noise_power / 2) # 最终噪声 = 标准噪声 * 目标标准差
    noise_imag = torch.randn_like(z.imag) * torch.sqrt(noise_power / 2)
    # 将独立的实部噪声和虚部噪声组合成一个复数噪声张量，用于添加到信号上
    noise = torch.complex(noise_real, noise_imag)

    return z + noise

# Rayleigh Fading Channel (slow fading)
def rayleigh_channel(z, snr_db):
    batch_size = z.shape[0] # 获取批次大小，为每一张图片成一个独立的衰落系数h
    power = torch.mean(torch.abs(z) ** 2) # 计算实际平均功率P
    snr_linear = 10 ** (snr_db / 10) # 将dB转换为线性比例, P / σ²
    noise_power = power / snr_linear # σ²
    
    # 生成复数衰落系数 h
    h_real = torch.randn(batch_size, 1, 1, 1, device=z.device) * (1 / math.sqrt(2)) # 实部的衰落系数
    h_imag = torch.randn(batch_size, 1, 1, 1, device=z.device) * (1 / math.sqrt(2)) # 虚部的衰落系数
    h = torch.complex(h_real, h_imag) # 将实部和虚部组合成复数衰落系数
    
    # 生成复数噪声
    noise_real = torch.randn_like(z.real) * torch.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(z.imag) * torch.sqrt(noise_power / 2)
    noise = torch.complex(noise_real, noise_imag)
    
    return h * z + noise

def resize_input(x, target_size=(32, 32)):# 将输入图像调整为32x32
    return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """ 计算峰值信噪比 (PSNR)。
    Args:
        img1 (np.ndarray): 原始图像。
        img2 (np.ndarray): 重建图像。
        data_range (float): 图像数据范围，默认为1.0。
    Returns:
        float: 计算得到的PSNR值。
    """
    return compare_psnr(img1, img2, data_range=data_range)

def ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0, channel_axis: int = -1, multichannel: bool = True) -> float:
    # 将 multichannel 参数传递给底层的 compare_ssim 函数
    """ 计算结构相似性指数 (SSIM)。
    Args:
        img1 (np.ndarray): 原始图像。
        img2 (np.ndarray): 重建图像。
        data_range (float): 图像数据范围，默认为1.0。
        channel_axis (int): 通道轴，默认为-1。
        multichannel (bool): 是否为多通道图像，默认为True。
    Returns:
        float: 计算得到的SSIM值。
    """
    return compare_ssim(img1, img2, data_range=data_range, channel_axis=channel_axis, multichannel=multichannel)

def calculate_importance_weights(model, data_batch, target_batch, device):
    """
    计算一个批次数据中，每个语义特征通道的重要性权重。
    采用最标准的PyTorch梯度计算方法。
    """
    model.eval()
    data_batch, target_batch = data_batch.to(device), target_batch.to(device)

    # 1. Set requires_grad=True
    data_batch.requires_grad = True

    # 2. 编码器提取特征。因为 data_batch 需要梯度，
    #    所以 features 也会自动成为计算图的一部分并“携带”梯度信息。
    features = model.encoder(data_batch)

    # 3. 将特征送入解码器得到logits。
    logits = model.decoder(features)

    # 4. 计算 ŷ*
    probabilities = F.softmax(logits, dim=1)
    y_star = torch.gather(probabilities, 1, target_batch.unsqueeze(-1)).squeeze()

    # 5. 计算梯度：∂ŷ*/∂features
    #    data_batch -> features -> logits -> y_star
    #    PyTorch反向追溯，计算出y_star对中间变量features的梯度。
    feature_gradients = autograd.grad(
        outputs=y_star.sum(), #导数的和 = 和的导数
        inputs=features,
        retain_graph=False  # 释放
    )[0]

    # 恢复requires_grad状态
    data_batch.requires_grad = False
    
    # 6. 计算每个通道的重要性权重 w_k
    weights_per_sample = feature_gradients.sum(dim=[2, 3])
    w = weights_per_sample.mean(dim=0)

    # 7. 排序并返回权重和对应的原始索引
    sorted_weights, sorted_indices = torch.sort(w, descending=True)
    
    return sorted_weights, sorted_indices

def calculate_stii(sorted_weights, M, ber, p0=0.5):
    """
    根据论文公式(8)，计算STII(η)值。

    Args:
        sorted_weights (torch.Tensor): 已按降序排列的重要性权重向量。
        M (int): 准备发送的特征通道数量。
        ber (float): Bit Error Rate
        p0 (float, optional): 固有可预测性概率。对于BPSK默认为0.5。

    Returns:
        float: 计算出的STII(η)分数。
    """
    # C 是总的特征通道数
    C = len(sorted_weights)
    
    # 确保 M 不会超过 C
    M = min(M, C)
    
    w = sorted_weights.detach().cpu().numpy()

    # 计算公式(8)的分子
    # 第一项：成功传输的语义信息量
    transmitted_importance = np.sum(w[:M]) * (1 - ber)
    
    # 第二项：未传输特征的固有可预测性
    untransmitted_importance = np.sum(w[M:])
    inherent_predictability = p0 * untransmitted_importance
    
    numerator = transmitted_importance + inherent_predictability

    # 计算公式(8)的分母：总的重要性
    denominator = np.sum(w)
    
    # 避免除以零的错误
    if denominator == 0:
        return 0.0

    stii_score = numerator / denominator
    return stii_score

# BER计算函数 (公式9) 
def calculate_ber_rayleigh(snr_db):
    """
    根据论文公式(9)计算瑞利信道下的BER。
    
    Args:
        snr_db (float): dB单位的信噪比。

    Returns:
        float: 计算出的比特错误率。
    """
    snr_linear = 10**(snr_db / 10.0)
    ber = 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))
    return ber