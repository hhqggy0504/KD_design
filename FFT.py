import numpy as np
from scipy.ndimage import uniform_filter


def frost_filter(image, window_size=3, K=1.0, eps=1e-6):
    """
    Frost滤波去除SAR图像相干斑

    参数:
        image: 输入二维SAR图像 (numpy数组)
        window_size: 滤波窗口大小 (奇数, 默认3)
        K: 控制指数衰减的敏感度参数 (默认1.0)
        eps: 防止除以零的小常数 (默认1e-6)

    返回:
        滤波后的图像 (numpy数组)
    """
    # 输入校验
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须为奇数")
    if image.ndim != 2:
        raise ValueError("输入必须为单通道二维图像")

    pad = window_size // 2
    image_pad = np.pad(image, pad, mode='reflect')  # 反射填充处理边界
    output = np.zeros_like(image)
    rows, cols = image.shape

    # 生成距离平方模板
    y, x = np.ogrid[-pad:pad + 1, -pad:pad + 1]
    distance_sq = x ** 2 + y ** 2

    # 预计算均值和方差
    mu = uniform_filter(image_pad, size=window_size, mode='reflect')
    image_sq = image_pad ** 2
    mu_sq = uniform_filter(image_sq, size=window_size, mode='reflect')
    var = mu_sq - mu ** 2

    # 主处理循环
    for i in range(rows):
        for j in range(cols):
            # 获取当前窗口统计量
            i_pad, j_pad = i + pad, j + pad
            window_mu = mu[i_pad, j_pad]
            window_var = var[i_pad, j_pad]

            # 计算自适应参数
            C_square = window_var / (window_mu ** 2 + eps)
            D = K * C_square

            # 生成权重矩阵并归一化
            weights = np.exp(-D * distance_sq)
            weights /= np.sum(weights)

            # 应用权重
            window = image_pad[i:i + window_size, j:j + window_size]
            output[i, j] = np.sum(weights * window)

    return output


def enhanced_frost_filter(image, window_size=5, K=1.5, gamma=1.2, eps=1e-6):
    """
    改进版Frost滤波：增强亮度并抑制相干斑

    参数改进:
        gamma: 亮度增强系数 (默认1.2)
        新增对数域处理流程
    """
    # 输入校验和预处理
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须为奇数")
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # 对数变换处理乘性噪声
    image_log = np.log(image + eps)  # 防止log(0)

    # 滤波处理流程
    pad = window_size // 2
    image_pad = np.pad(image_log, pad, mode='reflect')
    output_log = np.zeros_like(image_log)
    rows, cols = image_log.shape

    # 预计算统计量
    y, x = np.ogrid[-pad:pad + 1, -pad:pad + 1]
    distance_sq = x ** 2 + y ** 2

    mu = uniform_filter(image_pad, window_size)
    mu_sq = uniform_filter(image_pad ** 2, window_size)
    var = mu_sq - mu ** 2

    # 向量化改进
    for i in range(rows):
        for j in range(cols):
            i_pad, j_pad = i + pad, j + pad
            window_mu = mu[i_pad, j_pad]
            window_var = var[i_pad, j_pad]

            C_square = window_var / (window_mu ** 2 + eps)
            D = K * C_square

            weights = np.exp(-D * distance_sq)
            window = image_pad[i_pad - pad:i_pad + pad + 1, j_pad - pad:j_pad + pad + 1]
            output_log[i, j] = np.sum(weights * window) / np.sum(weights)

    # 后处理流程
    output = np.exp(output_log)  # 指数变换恢复亮度
    output = np.clip(output * gamma, 0, 255)  # 亮度增强

    return output.astype(np.uint8)


# 增强版使用示例
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    # 读取并预处理
    img = cv2.imread('./FUSAR1/FUSAR/bridge/A7_892.png', cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32)

    # 改进滤波处理
    enhanced_result = enhanced_frost_filter(img_float,
                                            window_size=7,
                                            K=1.8,
                                            gamma=1.5)

    # 显示对比
    plt.figure(figsize=(15, 6))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('111原始图像')
    plt.subplot(132), plt.imshow(frost_filter(img_float), cmap='gray'), plt.title('222原滤波')
    plt.subplot(133), plt.imshow(enhanced_result, cmap='gray'), plt.title('333增强亮度版')
    plt.show()