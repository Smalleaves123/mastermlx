from __future__ import annotations

import numpy as np


def _as_image(image):
    arr = np.asarray(image, dtype=float)
    if arr.ndim not in (2, 3):
        raise ValueError("image must have shape (H, W) or (H, W, C)")
    return arr


def rgb_to_gray(image):
    image = _as_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] < 3:
        raise ValueError("RGB image must have at least 3 channels")
    return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]


def normalize_image(image, eps=1e-12):
    image = _as_image(image)
    lo = float(np.min(image))
    hi = float(np.max(image))
    if hi - lo < eps:
        return np.zeros_like(image, dtype=float)
    return (image - lo) / (hi - lo)


def resize_nearest(image, size):
    image = _as_image(image)
    h, w = map(int, size)
    if h < 1 or w < 1:
        raise ValueError("size must be positive")
    src_h, src_w = image.shape[:2]
    y_idx = np.minimum((np.linspace(0, src_h - 1, h)).round().astype(int), src_h - 1)
    x_idx = np.minimum((np.linspace(0, src_w - 1, w)).round().astype(int), src_w - 1)
    return image[np.ix_(y_idx, x_idx)] if image.ndim == 2 else image[np.ix_(y_idx, x_idx, np.arange(image.shape[2]))]


def integral_image(image):
    image = _as_image(image)
    if image.ndim == 3:
        image = rgb_to_gray(image)
    return np.cumsum(np.cumsum(image, axis=0), axis=1)


def box_blur(image, ksize=3):
    image = _as_image(image)
    if ksize < 1 or int(ksize) != ksize:
        raise ValueError("ksize must be a positive integer")
    ksize = int(ksize)
    pad = ksize // 2
    if image.ndim == 3:
        channels = [box_blur(image[..., c], ksize=ksize) for c in range(image.shape[2])]
        return np.stack(channels, axis=2)
    padded = np.pad(image, pad_width=pad, mode="edge")
    out = np.empty_like(image, dtype=float)
    area = float(ksize * ksize)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i : i + ksize, j : j + ksize]) / area
    return out


def sobel_edges(image):
    image = _as_image(image)
    if image.ndim == 3:
        image = rgb_to_gray(image)
    padded = np.pad(image, pad_width=1, mode="edge")
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    grad_x = np.empty_like(image, dtype=float)
    grad_y = np.empty_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i : i + 3, j : j + 3]
            grad_x[i, j] = np.sum(window * gx)
            grad_y[i, j] = np.sum(window * gy)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (n, 4)")
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes and scores must have the same length")
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = max(0.0, (boxes[i, 2] - boxes[i, 0])) * max(0.0, (boxes[i, 3] - boxes[i, 1]))
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        iou = np.divide(inter, np.maximum(union, 1e-12))
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=int)


def sliding_window(image, window_size, step=1):
    image = _as_image(image)
    if image.ndim == 3:
        image = rgb_to_gray(image)
    win_h, win_w = map(int, window_size)
    step = int(step)
    if win_h < 1 or win_w < 1 or step < 1:
        raise ValueError("window_size and step must be positive")
    if image.shape[0] < win_h or image.shape[1] < win_w:
        raise ValueError("window_size must fit within the image")
    windows = []
    boxes = []
    for top in range(0, image.shape[0] - win_h + 1, step):
        for left in range(0, image.shape[1] - win_w + 1, step):
            boxes.append((left, top, left + win_w, top + win_h))
            windows.append(image[top : top + win_h, left : left + win_w])
    return np.asarray(boxes, dtype=float), np.asarray(windows, dtype=float)


def generate_proposals(image, window_sizes=((16, 16), (32, 32)), step=8, score_fn=None, iou_threshold=0.5):
    """Generate simple detection proposals with optional scoring and NMS."""

    image = _as_image(image)
    if score_fn is None:
        def score_fn(patch):
            return float(np.mean(patch))
    all_boxes = []
    all_scores = []
    for window_size in window_sizes:
        boxes, patches = sliding_window(image, window_size, step=step)
        scores = np.array([score_fn(patch) for patch in patches], dtype=float)
        all_boxes.append(boxes)
        all_scores.append(scores)
    boxes = np.concatenate(all_boxes, axis=0) if all_boxes else np.empty((0, 4), dtype=float)
    scores = np.concatenate(all_scores, axis=0) if all_scores else np.empty((0,), dtype=float)
    keep = non_max_suppression(boxes, scores, iou_threshold=iou_threshold) if boxes.size else np.array([], dtype=int)
    return boxes[keep], scores[keep]
