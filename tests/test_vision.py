import numpy as np

from mastermlx.vision import (
    generate_proposals,
    box_blur,
    histogram_of_oriented_gradients,
    integral_image,
    normalize_image,
    non_max_suppression,
    patch_descriptor,
    resize_nearest,
    sliding_window,
    rgb_to_gray,
    sobel_edges,
)


def test_rgb_to_gray_and_normalize():
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    gray = rgb_to_gray(image)
    norm = normalize_image(gray)
    assert gray.shape == (1, 2)
    assert norm.min() == 0.0
    assert norm.max() == 1.0


def test_resize_blur_and_integral_image():
    image = np.arange(16, dtype=float).reshape(4, 4)
    resized = resize_nearest(image, (2, 2))
    blurred = box_blur(image, ksize=3)
    integ = integral_image(image)
    assert resized.shape == (2, 2)
    assert blurred.shape == (4, 4)
    assert np.isclose(integ[-1, -1], np.sum(image))


def test_sobel_and_nms():
    image = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    mag, direction = sobel_edges(image)
    boxes = np.array([[0, 0, 2, 2], [0, 0, 1.5, 1.5], [2, 2, 3, 3]])
    scores = np.array([0.7, 0.9, 0.6])
    keep = non_max_suppression(boxes, scores, iou_threshold=0.3)
    assert mag.shape == (3, 3)
    assert direction.shape == (3, 3)
    assert set(keep.tolist()) == {1, 2}


def test_sliding_window_and_proposals():
    image = np.arange(25, dtype=float).reshape(5, 5)
    boxes, patches = sliding_window(image, (3, 3), step=2)
    proposals, scores = generate_proposals(image, window_sizes=((3, 3),), step=2)
    assert boxes.shape == (4, 4)
    assert patches.shape == (4, 3, 3)
    assert proposals.shape[1] == 4
    assert scores.ndim == 1
    assert proposals.shape[0] <= boxes.shape[0]


def test_hog_and_patch_descriptor():
    image = np.arange(64, dtype=float).reshape(8, 8)
    hog = histogram_of_oriented_gradients(image, cell_size=4, bins=6, block_size=2)
    patch = patch_descriptor(image, patch_size=(4, 4))
    assert hog.ndim == 1
    assert hog.size > 0
    assert patch.shape == (16,)
    assert np.isclose(np.linalg.norm(patch), 1.0)
