from __future__ import division
from PIL import Image
import numpy as np

class EM:
    def __init__(self, file_name):
        self.img = Image.open(file_name)
        self.pixels = self.img.load()

        data = []
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                data.append(self.pixels[i, j])

        self.data = np.array(data)

if __name__ == '__main__':
    em = EM("test_images/nature.jpg")