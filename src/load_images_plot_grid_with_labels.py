import math
import os
import pathlib
from glob import glob

import matplotlib.pyplot as plt
from IPython.display import Image

# input_dir = "/media/hdd/Datasets/treecover_segmentation/augmentedtiles"
# images = os.listdir(input_dir)
# images = list(set([x.split("_")[0] for x in images]))
# images = [input_dir +"/"+ x + "_1.tif" for x in images]
images = os.listdir("outputs/")
images = ["outputs/" + x for x in images]
images.sort()
print(images)
# exit(0)
col = 2
image_count = len(images)
row = math.ceil(image_count / col)
plt.figure(figsize=(col * 4, row * 4))
plt.figure(figsize=(col * 4, row * 4))

for i, img_path in enumerate(images):
    img_path = str(img_path)

    img = plt.imread(img_path)

    plt.subplot(row, col, i + 1)
    if i % 2 == 0:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title(pathlib.Path(img_path).name)
    plt.axis("off")

# plt.show()
# plt.savefig("augmentedtiles.pdf")
plt.savefig("results.pdf")
plt.close()
