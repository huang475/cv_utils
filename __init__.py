# %%
from os import walk, path, remove
from timeit import default_timer as timer
import inspect

import numpy as np
import matplotlib.pyplot as plt
import cv2


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
        cv2.rectangle(ret, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 3)

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
    boxA = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3]/2, bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    boxB = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3]/2, bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2

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
