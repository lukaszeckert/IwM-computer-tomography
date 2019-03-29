import numpy
# import pyximport
import pyximport;

pyximport.install(pyximport=True, setup_args={'include_dirs': [numpy.get_include(), "."]})
from _tomograph import *


class ComputerTomography:
    def __init__(self, number_steps, number_emitters, scan_width):
        self.number_steps = number_steps
        self.number_emitters = number_emitters
        self.scan_width = scan_width

    def create_filter(self, size):
        res = np.zeros(size)
        res[int(size / 2)] = 1.0
        for i in range(size // 2 + 1, size):
            if i % 2 == size / 2 % 2:
                res[2 * (size // 2) - i], res[i] = 0.0, 0.0
            else:
                value = -4 / np.pi ** 2 / (i-size//2) ** 2
                res[2 * (size // 2) - i], res[i] = value, value
        return res

    def radon_transform(self, img: np.array, filter=False):
        res = np.asanyarray(
            radon_transform(img.astype(np.float), self.number_steps, self.number_emitters, self.scan_width))
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        if filter:

            number_steps = res.shape[0]
            number_emitters = res.shape[1]
            kernel = self.create_filter(number_emitters)
            for i in range(number_steps):
                res[i, :] = convolve(res[i], kernel, res[i].shape[0], kernel.shape[0])
        return res

    def radon_transform_iterative(self, img: np.array, filter=False, update_fun=None):
        res = np.zeros((self.number_steps, self.number_emitters))
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        if self.number_emitters%2:
            kernel = self.create_filter(self.number_emitters+1)
        else:
            kernel = self.create_filter(self.number_emitters)

        for i in range(self.number_steps):
            res[i, :] = radon_transform_iterative(img.astype(np.float), self.number_steps, self.number_emitters,
                                                  self.scan_width, i)
            if filter:
                res[i, :] = convolve(res[i], kernel, res[i].shape[0], kernel.shape[0])
            if i%10 == 0:
                update_fun(res)

        update_fun(res)

    def iradon_transform(self, img: np.array):
        res = np.asanyarray(
            iradon_transform(img.astype(np.float), self.number_steps, self.number_emitters, self.img_width,
                             self.img_height, self.scan_width))
        res = (res - np.min(res)) / np.max(res)
        return res

    def iradon_transform_iterative(self, img: np.array, update_fn):
        res = np.zeros((self.img_height,self.img_width))
        for i in range(img.shape[0]):
            res += iradon_transform_iterative(img.astype(np.float), self.number_steps, self.number_emitters,
                                                  self.img_width,self.img_height, self.scan_width, i)
            #print(tmp.shape)
            if i % 100 == 0:
                tmp = (res - np.min(res)) / np.max(res)
                update_fn(tmp)
        tmp = (res - np.min(res)) / np.max(res)
        update_fn(tmp)
