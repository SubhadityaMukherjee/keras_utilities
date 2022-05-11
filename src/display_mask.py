import PIL
import numpy as np
from tensorflow import keras
from IPython.display import Image, display


def display_mask(val_preds, i, fname=None):
    """Quick utility to display a model's prediction. It takes an image and generates a mask from it."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    if fname != None:
        img.save(fname)
    display(img)
