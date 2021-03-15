from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern as LBP
import numpy as np

def Felzenszwalb(img, scale, sigma, min_size):
    # img: [h, w, 3]
    # mask: [h, w]
    mask = felzenszwalb(img_as_float(img), scale=scale, sigma=sigma, min_size=min_size)
    # mask_layer: [h, w, 1]
    mask_layer = np.zeros(img.shape[:2])[:, :, np.newaxis]
    img_mask = np.concatenate([img, mask_layer], axis=2)
    img_mask[:, :, 3] = mask
    return img_mask

# 使用局部二值模式算法提取纹理特征
def texture_feature(img):
    texture = np.zeros(*r.shape)
    for c in range(3):
        # P: 选择中心像素周围像素点数
        # R: 选择像素距离中心像素的最大半径
        # R=8, R=1: 选择中心像素8个方向各一个像素点
        texture[:, :, c] = LBP(img, P=8, R=1)
    return texture

# 计算纹理特征频数
def texture_hist(texture, bins=10):
    # texture: [h * w, 4]
    hist = []
    for c in range(3):
        hist.append(np.histogram(texture[:, c], bins=bins)[0])
    # hist: [3 * bins]
    hist = np.concatenate(hist)
    # L1 normalize
    hist = hist / texture.shape[0]
    return hist

# 计算颜色特征频数
def color_hist(img, bins=25):
    # img: [h * w, 4]
    hist = []
    for c in range(3):
        hist.append(np.histogram(img[:, c], bins=bins)[0])
    # hist: [3 * bins]
    hist = np.concatenate(hist)
    # L1 normalize
    hist = hist / img.shape[0]
    return hist

def get_R(img_mask):
    R = {}
    # img_mask: [h, w, 4]
    # 遍历每一个像素, 并根据 mask 进行归类
    for y, w4 in enumerate(img_mask):
        for x, (r, g, b, mask) in enumerate(w4):
            if mask not in R:
                # 将 x_min, y_min 设置为最大值，将 x_max, y_max 设置为最小值, 以便后续的比较
                # mask 用于标识像素点是否属于同一区域
                R[mask] = {
                    "min_x": 0xffff, "max_x": 0, "min_y": 0xffff, "max_y": 0,
                    "region": mask
                }
            if R[mask]["min_x"] > x:
                R[mask]["min_x"] = x
            if R[mask]["max_x"] < x:
                R[mask]["max_x"] = x
            if R[mask]["min_y"] > y:
                R[mask]["min_y"] = y
            if R[mask]["max_y"] < y:
                R[mask]["max_y"] = y
    return R