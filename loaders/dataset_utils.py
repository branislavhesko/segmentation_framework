import os
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np


def get_dataset(dataset_path, extensions, mode):
    assert mode in ["train", "validate", "test"]
    assert len(extensions) == 2

    images = glob.glob(os.path.join(dataset_path, mode, "imgs")+ "/*." + extensions[0])
    masks = glob.glob(os.path.join(dataset_path, mode, "masks")+ "/*." + extensions[1])

    assert len(images) == len(masks)

    data = [(image, mask) for image, mask in zip(images, masks)]
    return data


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


if __name__ == '__main__':
    N = 5

    x = np.random.randn(40)
    y = np.random.randn(40)
    c = np.random.randint(N, size=40)

    # Edit: don't use the default ('jet') because it makes @mwaskom mad...
    plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, 'cubehelix'))
    plt.colorbar(ticks=range(N))
    plt.clim(-0.5, N - 0.5)
    plt.show()