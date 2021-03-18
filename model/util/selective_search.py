from skimage.util import img_as_float
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern as LBP
import numpy as np

# 基于图的图像分割
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
    texture = np.zeros(img.shape)
    for c in range(3):
        # P: 选择中心像素周围像素点数
        # R: 选择像素距离中心像素的最大半径
        # R=8, R=1: 选择中心像素8个方向各一个像素点
        texture[:, :, c] = LBP(img[:, :, c], P=8, R=1)
    return texture

# 计算纹理特征频数
def texture_hist(texture, bins=10):
    # texture: [<=(h * w), 4]
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
    # img: [<=(h * w), 4]
    hist = []
    for c in range(3):
        hist.append(np.histogram(img[:, c], bins=bins)[0])
    # hist: [3 * bins]
    hist = np.concatenate(hist)
    # L1 normalize
    hist = hist / img.shape[0]
    return hist

# 获取R集
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
                    "region": [mask]
                }
            if R[mask]["min_x"] > x:
                R[mask]["min_x"] = x
            if R[mask]["max_x"] < x:
                R[mask]["max_x"] = x
            if R[mask]["min_y"] > y:
                R[mask]["min_y"] = y
            if R[mask]["max_y"] < y:
                R[mask]["max_y"] = y
    # 提取图像区域纹理特征
    texture = texture_feature(img_mask)
    for k, v in list(R.items()):
        # 提取出每个通道中符合掩码的纹理特征值
        # [h, w, 4] => [<=(h * w), 4]
        texture_mask = texture[img_mask[:, :, 3] == k]
        # 存储图像区域大小
        R[k]["size"] = texture_mask.shape[0]
        # 存储图像区域纹理特征频数
        R[k]["hist_t"] = texture_hist(texture_mask)
        # 提取出每个通道中符合掩码的像素颜色特征
        # [h, w, 4] => [<=(h * w), 4]
        color_mask = img_mask[img_mask[:, :, 3] == k]
        # 存储图像区域颜色特征频数
        R[k]["hist_c"] = color_hist(color_mask)
    return R

# 计算相似度
def similarity(r1, r2, img_size):
    # 纹理相似度
    texture_sim = 0
    for r1_ht, r2_ht in zip(r1["hist_t"], r2["hist_t"]):
        texture_sim += min(r1_ht, r2_ht)
    # 颜色相似度
    color_sim = 0
    for r1_hc, r2_hc in zip(r1["hist_c"], r2["hist_c"]):
        color_sim += min(r1_hc, r2_hc)
    # 大小相似度
    size_sim = 1 - (r1["size"] + r2["size"]) / img_size
    # 填充相似度
    w_box = (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
    h_box = (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    fill_sim = 1 - (w_box * h_box - r1["size"] - r2["size"]) / img_size
    return texture_sim + color_sim + size_sim + fill_sim

# 判断图像区域是否相邻
def isneighbor(r1, r2):
    # r2 在 r1 的左上角
    left_top = (r1["min_x"] <= r2["max_x"] <= r1["max_x"]) and (r1["min_y"] <= r2["max_y"] <= r1["max_y"])
    # r2 在 r1 的右上角
    right_top = (r1["min_x"] <= r2["min_x"] <= r1["max_x"]) and (r1["min_y"] <= r2["max_y"] <= r1["max_y"])
    # r2 在 r1 的左下角
    left_bottom = (r1["min_x"] <= r2["max_x"] <= r1["max_x"]) and (r1["min_y"] <= r2["min_y"] <= r1["max_y"])
    # r2 在 r1 的右下角
    right_bottom = (r1["min_x"] <= r2["min_x"] <= r1["max_x"]) and (r1["min_y"] <= r2["min_y"] <= r1["max_y"])
    return left_top or right_top or left_bottom or right_bottom

# 获取相邻区域对
def neighbors(R):
    R = list(R.items())
    N = []
    # 遍历所有区域（除了最后一个）r1
    for i, (k1, r1) in enumerate(R[:-1]):
        # 遍历位于 r1 之后的所有区域
        for k2, r2 in R[i + 1:]:
            if isneighbor(r1, r2):
                N.append([(k1, r1), (k2, r2)])
    return N

# 合并区域
def merge(r1, r2):
    r_new = {"min_x": min(r1["min_x"], r2["min_x"]), "max_x": max(r1["max_x"], r2["max_x"]),
             "min_y": min(r1["min_y"], r2["min_y"]), "max_y": max(r1["max_y"], r2["max_y"]),
             "size": r1["size"] + r2["size"], "region": r1["region"] + r2["region"]}
    r_new["hist_t"] = (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / r_new["size"]
    r_new["hist_c"] = (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / r_new["size"]
    return r_new

# 选择搜索算法
def selective_search(img, scale=1, sigma=0.8, min_size=50):
    img_size = img.shape[0] * img.shape[1]
    # 图像分割
    img_mask = Felzenszwalb(img, scale, sigma, min_size)
    R = get_R(img_mask)
    # 初始化 S 集
    S = {}
    for (k1, r1), (k2, r2) in neighbors(R):
        S[(k1, k2)] = similarity(r1, r2, img_size)
    while S:
        print(f"R:{len(R)} S:{len(S)}")
        # 查找到相似度最高的两个区域
        k1, k2 = max(list(S.items()), key=lambda x:x[1])[0]
        # 合并生成新区域 r_new, 并将其存入 R 集
        r_new = merge(r1, r2)
        k_new = max(R.keys()) + 1
        R[k_new] = r_new
        # 从 S 集查找与 r1, r2 有关的区域
        related = []
        for k, v in list(S.items()):
            if (k1 in k) or (k2 in k):
                related.append(k)
        # 从 S 集删除与 r1, r2 有关的相似度
        for k in related:
            S.pop(k)
        # 与 r1, r2 相邻和区域也会与 r_new 相邻, 计算新的相似度加入 S 集
        for k in [k for k in related if k != (k1, k2)]:
            # k_other 为与 r1, r2 相邻的区域的 key
            k_other = k[1] if k[0] in (k1, k2) else k[0]
            S[(k_new, k_other)] = similarity(R[k_new], R[k_other], img_size)
    candidate_regions = []
    for k, v in list(R.items()):
        candidate_regions.append({
            "box": (v["min_x"], v["min_y"], v["max_x"] - v["min_x"], v["max_y"] - v["min_y"]),
            "size": v["size"],
            "region": v["region"]
        })
    return candidate_regions

if __name__ == "__main__":
    import cv2
    img = cv2.imread("demo.jpg")
    r = selective_search(img)
    print(r)