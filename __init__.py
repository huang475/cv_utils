# %%
from os import walk, path, remove
from timeit import default_timer as timer
import inspect
from functools import reduce
from random import random

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def filenames_in_dir(d):
    filenames = []
    for _, _, _filenames in walk(d):
        filenames += (_filenames)
    return filenames


def passert(func):
    import inspect
    print("asserting {}".format(
        inspect.getsource(func)
    ))


def annotate_image_with_boxes(frame, bboxes):
    ret = frame[:]

    img_h, img_w, _ = frame.shape

    for x, y, w, h, p in bboxes:
        cv2.rectangle(ret, (int(x - w/2), int(y - h/2)),
                      (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)

    return ret


def annotate_image_with_points(image, points):
    ret = image[:]
    for x, y in points:
        cv2.circle(ret, (x, y), 3, (0, 0, 255), 3)
    return ret


def visualize_points(points, size, name=None):
    img = np.ones(size)
    img = annotate_image_with_points(img, points)
    visualize(**{"{}".format(name or "image"): img})


def iou(bbox1, bbox2):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3] / \
        2, bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    boxB = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3] / \
        2, bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    ret = interArea / float(boxAArea + boxBArea - interArea)

    return ret


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(12, 4))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image.squeeze())
    plt.show()


def measure(func, *args, **kwargs):
    # print("measuring {}".format(inspect.cleandoc(func)))
    start = timer()
    r = func(*args, **kwargs)
    end = timer()
    diff = end - start
    print("execution take {} seconds".format(diff))
    return r


def split_dataset(imagenames, file_paths, scales):
    if isinstance(imagenames, str):
        imagenames = [path.join(imagenames, f) for f in filenames_in_dir(
            imagenames) if f.split('.')[-1] in ['png', 'jpg', 'jpeg']]

    assert len(file_paths) == len(
        scales), "{} and {} should have same length".format(file_paths, scales)
    assert 0.99 < scales[-1] < 1.01, "last of scales should be 1, scales provided: {}".format(
        scales)
    try:
        files = [open(p, 'w') for p in file_paths]
        for imagename in imagenames:
            dice = random()
            for f, scale in zip(files, scales):
                if scale > dice:
                    f.write("{}\n".format(imagename))
                    break
    finally:
        for f in files:
            f.close()


def generate_dataset_file(dataset_file_path, num_of_classes, train_set_file_path, validation_set_file_path, names_file_path):
    with open(dataset_file_path, 'w') as f:
        f.write("classes={}\n".format(num_of_classes))
        f.write("train={}\n".format(train_set_file_path))
        f.write("valid={}\n".format(validation_set_file_path))
        f.write("names={}\n".format(names_file_path))


def num_of_nonempty_lines(file_path):
    num_of_lines = 0
    with open(file_path) as f:
        lines = [f for f in f.readlines() if len(f.strip()) > 0]
        num_of_lines = len(lines)
    return num_of_lines


def read_bboxes_from_yolo_annotation_path(label_path):
    files = [f for f in filenames_in_dir(label_path)
             if f.endswith('txt')]
    bboxes = []
    for filename in files:
        with open(path.join(label_path, filename)) as f:
            _bboxes = [[float(s) for s in line[:-1].split(" ")[1:5]]
                       for line in f.readlines()
                       if line.strip()
                       ]
            bboxes += _bboxes

    return bboxes


def bbox_classify(bboxes, possible_k):
    """bbox: x, y, w, h
    return: best kmeans score anchor classes [(w1, h1), (w2, h2), ...]
    """
    anchors = [bbox[2:4] for bbox in bboxes]
    return anchors_classify(anchors, possible_k)


def anchors_classify(anchors, possible_k):
    if isinstance(possible_k, int):
        possible_k = [possible_k]
    assert all([k > 1 for k in possible_k]
               ), "k must be larger than 1, got: {}".format(possible_k)
    assert len(possible_k) > 1, "must provide at least one k value"

    best_k = 0
    best_k_score = 0
    best_kmeans = []


    for k in possible_k:
        kmeans = KMeans(n_clusters=k).fit(anchors)
        score = silhouette_score(anchors, kmeans.labels_, metric='euclidean')
        if score > best_k_score:
            best_k = k
            best_k_score = score
            best_kmeans = kmeans
    return best_kmeans.cluster_centers_, best_k_score


def find_best_anchors_from_annotations_files(file_path, possible_num_of_anchors, image_width, image_height):
    bboxes = read_bboxes_from_yolo_annotation_path(file_path)
    anchors, _ = bbox_classify(bboxes, possible_num_of_anchors)
    return [(int(image_width * anchor[0]),
             int(image_height * anchor[1]))
            for anchor in anchors]
