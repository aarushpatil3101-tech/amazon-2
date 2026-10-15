"""Image -> numpy feature helpers for the classical baselines."""
import os

import numpy as np
from PIL import Image


def load_images(ids, img_dir, size=64, gray=False):
    """Load images as a uint8 array (N, size, size, 3) or (N, size, size) if gray."""
    shape = (len(ids), size, size) if gray else (len(ids), size, size, 3)
    out = np.empty(shape, dtype=np.uint8)
    for k, i in enumerate(ids):
        im = Image.open(os.path.join(img_dir, f"{i}.jpg")).convert("L" if gray else "RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out[k] = np.asarray(im)
    return out


def hog_features(gray_imgs, pixels_per_cell=8, cells_per_block=2, orientations=9):
    from skimage.feature import hog
    feats = [hog(g, orientations=orientations, pixels_per_cell=(pixels_per_cell,) * 2,
                 cells_per_block=(cells_per_block,) * 2, feature_vector=True) for g in gray_imgs]
    return np.asarray(feats, dtype=np.float32)
