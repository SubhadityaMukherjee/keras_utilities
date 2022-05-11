import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_images_to_array(dss_path):
    """
    Return images and labels as numpy arrays
    """
    images, labels = [], []
    for root, dirs, files in tqdm(os.walk(dss_path)):
        for file in files:
            if file.endswith(".pgm"):
                fname = os.path.join(root, file)
                images.append(np.array(Image.open(fname).convert("L").resize(
                    (28, 28), Image.Resampling.BILINEAR)))  # Biliear is good sampling
                # print(images.shape)
                # get the last folder name as label
                labels.append(root.split("/")[-1])
    return images, labels


def label_to_dict(labels):
    labelmap = {label: i for i, label in enumerate(np.unique(labels))}
    return labelmap, [labelmap[label] for label in labels]
