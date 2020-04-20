from os import path, listdir, remove
from functools import reduce

import pandas as pd
import numpy as np
import PIL.Image as Image

def generate_names(vott_annotation_file, names_file):
    annotations = read_annotations(vott_annotation_file)
    labels = sorted(list(set([d['label'] for d in annotations])))
    with open(names_file, 'w') as f:
        f.write("\n".join(labels))

def read_names_from_file(names_file):
    names = []
    with open(names_file, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    return names

def read_names_with_index_from_file(names_file):
    name_index_dict = {}
    names = read_names_from_file(names_file)
    for idx, name in enumerate(names):
        name_index_dict[name] = idx
    return name_index_dict

def get_annotation_file_path(dir_path, image_name):
    return path.join(dir_path, '.'.join(image_name.split('.')[:-1]) + '.txt')

def generate_yolo_annotaitons(vott_annotation_file, yolo_names_file, images_dir, output):
    annotations = read_annotations(vott_annotation_file)
    names = read_names_from_file(yolo_names_file)
    group_by_image = {}

    for image_name in [f for f in listdir(images_dir) if path.isfile(path.join(images_dir, f))]:
        group_by_image[image_name] = []

    # group by image name
    for annotation in annotations:
        image_name = annotation['image']
        assert annotation['label'] in names, "{} not in names".format(annotation['label'])
        if image_name in group_by_image:
            group_by_image[image_name].append(annotation)
        else:
            group_by_image[image_name] = [annotation]

    # generate annotation file for each image
    for image_name, annotations in group_by_image.items():
        image_file_path = path.join(images_dir, image_name)
        annotation_file_path = get_annotation_file_path(output, image_name)


        if len(annotations) == 0:
            if path.exists(image_file_path):
                remove(image_file_path)

            if path.exists(annotation_file_path):
                remove(annotation_file_path)

            continue

        img = Image.open(path.join(images_dir, image_name))

        vott_labels = np.array([
            [1, a['xmin'], a['xmax'], a['ymin'], a['ymax'], names.index(a['label'])] 
            for a in annotations
            ])
        width, height = img.size

        # calculate yolo anchor format (x, y, w, h) from vott box format (x1, x2, y1, y2)
        targets = np.zeros(vott_labels.shape, dtype=np.float32)
        targets[:, 0] = 0
        targets[:, 1] = (
            vott_labels[:, 1] + (vott_labels[:, 2] - vott_labels[:, 1]) / 2) / width
        targets[:, 2] = (
            vott_labels[:, 3] + (vott_labels[:, 4] - vott_labels[:, 3]) / 2) / height
        targets[:, 3] = (vott_labels[:, 2] - vott_labels[:, 1]) / width
        targets[:, 4] = (vott_labels[:, 4] - vott_labels[:, 3]) / height
        targets[:, 5] = vott_labels[:, 5]

        with open(annotation_file_path, 'w') as f:
            labels = [ "{} {} {} {} {}".format(
                int(targets[i][0]), targets[i][1], targets[i][2], targets[i][3], targets[i][4]
            ) for i in range(targets.shape[0])]
            f.write("\n".join(labels))


def read_annotations(vott_annotation_file):
    # annotation fields: image, xmin, xmax, ymin, ymax, label
    annotations = pd.read_csv(vott_annotation_file).to_dict(orient='records')
    return annotations


if __name__ == "__main__":
    p = "/home/yuchen/workspace/data/huarun/worker_detection/"
    img_dir_path = p
    vott_annotation_file = path.join(p, "39_worker_detection-export.csv")
    output_d = path.join(p, "yolo_annotations")
    names_f = path.join(p, "39_worker_detection.names")
    generate_names(vott_annotation_file, names_f)
    print(read_names_from_file(names_f))
    generate_yolo_annotaitons(vott_annotation_file, names_f, img_dir_path, output_d)

