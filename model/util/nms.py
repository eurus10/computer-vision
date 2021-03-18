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
        delete = []
        for j in index:
            if IoU(det, i, j) >= theta:
                delete.append(j)
        for j in delete:
            index.remove(j)
    return det[keep]

if __name__ == "__main__":
    import cv2
    import torch
    img = cv2.imread("1.png")
    det = torch.tensor([[80, 280, 30, 40, 0.9],
                        [82, 278, 32, 45, 0.8],
                        [77, 281, 30, 38, 0.6],
                        [260, 270, 30, 60, 0.7],
                        [254, 273, 34, 62, 0.8]])
    for d in det:
        img = cv2.rectangle(img, (d[0] - d[2] / 2, d[1] - d[3] / 2), (d[0] + d[2] / 2, d[1] + d[3] / 2), (255, 255, 0), 1)
    cv2.imwrite("1_.png", img)
    img = cv2.imread("1.png")
    for d in nms(det, 0.5):
        img = cv2.rectangle(img, (d[0] - d[2] / 2, d[1] - d[3] / 2), (d[0] + d[2] / 2, d[1] + d[3] / 2), (255, 255, 0), 1)
    cv2.imwrite("1_nms.png", img)