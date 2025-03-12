import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_random_image_from_folder(folder_path):
    """从文件夹中随机选择一张图片"""
    images = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
    if not images:
        return None
    random_image = random.choice(images)
    return os.path.join(folder_path, random_image)


def remove_noise(img):
    """使用高斯模糊去除图像噪声"""
    return cv2.GaussianBlur(img, (5, 5), 0)


def extract_frequency_features(magnitude_spectrum):
    """提取频域特征，包括低频和高频信息"""
    h, w = magnitude_spectrum.shape
    center_x, center_y = w // 2, h // 2

    low_freq_region = magnitude_spectrum[center_y - 10:center_y + 10, center_x - 10:center_x + 10]
    high_freq_region = magnitude_spectrum[:10, :10]  # 取左上角的一部分代表高频

    features = {
        "mean": np.mean(magnitude_spectrum),
        "std": np.std(magnitude_spectrum),
        "energy": np.sum(magnitude_spectrum ** 2),
        "entropy": -np.sum(magnitude_spectrum * np.log(magnitude_spectrum + 1)),
        "low_freq_mean": np.mean(low_freq_region),
        "high_freq_mean": np.mean(high_freq_region)
    }
    return features


def convert_to_frequency_domain(image_path):
    """将图像转换到频域，并去除噪声"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None

    denoised_img = remove_noise(img)

    f = np.fft.fft2(denoised_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # 避免log(0)

    features = extract_frequency_features(magnitude_spectrum)

    return denoised_img, magnitude_spectrum, features


def main():
    base_folder = "FUSAR1/FUSAR"
    subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if
                  os.path.isdir(os.path.join(base_folder, d))]

    if len(subfolders) < 7:
        print("文件夹数量不足7个！")
        return

    selected_subfolders = subfolders[:7]  # 选择前7个文件夹

    plt.figure(figsize=(14, 7))

    for i, folder in enumerate(selected_subfolders):
        image_path = get_random_image_from_folder(folder)
        if not image_path:
            print(f"{folder} 没有找到图片！")
            continue

        img, spectrum, features = convert_to_frequency_domain(image_path)
        if img is None or spectrum is None:
            print(f"无法处理 {image_path}")
            continue

        print(f"Features for {os.path.basename(image_path)}: {features}")

        plt.subplot(7, 2, i * 2 + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Denoised Image: {os.path.basename(image_path)}")
        plt.axis("off")

        plt.subplot(7, 2, i * 2 + 2)
        plt.imshow(spectrum, cmap='gray')
        plt.title("Frequency Spectrum")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
