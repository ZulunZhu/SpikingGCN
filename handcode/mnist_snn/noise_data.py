import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import util

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

kind_list = ['train', 'test']
data_list = ['0']
noise = 'gaussian'

fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

for data in data_list:
    os.mkdir('./' + noise + '_mean' + data)

for data in data_list:
    for kind in kind_list:
        path = './' + noise + '_mean' + data + '/'
        img, labels = load_mnist('./mnist/raw', kind = kind)
        # img = util.random_noise(img, mode = noise, seed = 3154, mean = float(data))
        img = util.random_noise(img, mode = 's&p', seed = 3154, amount = float(data))
        img = img * 255
        images = img.astype(np.int)

        ax = ax.flatten()
        for i in range(10):
            img = images[labels == i][0].reshape(28, 28)
            ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig(path + kind + '.png')

        np.savetxt(path + kind + '_images.csv', images, fmt = '%i', delimiter = ',')
        np.savetxt(path + kind + '_labels.csv', labels, fmt = '%i', delimiter = ',')