def IoU(det, i, j):
    # 提取 bounding box 信息
    x = det[[i, j], 0]
    y = det[[i, j], 1]
    w = det[[i, j], 2]
    h = det[[i, j], 3]
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    size = w * h
    inter_x_min = max(x_min)
    inter_x_max = min(x_max)
    inter_y_min = max(y_min)
    inter_y_max = min(y_max)
    inter_size = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    return inter_size / (sum(size) - inter_size)

# det: [[x_center, y_center, width, height, classes ... ], ...]
# theta: IoU 阙值
def nms(det, theta):
    prob = det[:, 4]
    index = prob.argsort().tolist() # 剩余索引, 按可能性从小到大排序
    keep = [] # 需要保留的索引
    while index:
        # 将可能性最高的留下
        i = index[-1]
        index.remove(i)
        keep.append(i)
        for j in index:
            if IoU(det, i, j) >= theta:
                index.remove(j)
    return det[keep]

if __name__ == "__main__":
    import torch
    det = torch.tensor([[2, 2, 2, 2, 0.9], [2, 2, 2, 4, 0.8], [3, 1, 2, 2, 0.6], [6, 2, 2, 2, 0.5]])
    result = nms(det, 0.5)
    print(result.tolist())