'''
cs498 mixture_normal
Created on 3/15/16
@author: xiaofo
'''

from __future__ import division
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class GaussianEM:

    def __init__(self,file_name,segment_num=10):
        # initialize data
        self.image = Image.open(file_name)
        self.pixels = self.image.load()
        self.width = self.image.size[0]
        self.height = self.image.size[1]
        self.pixel_num = self.width * self.height
        self.l = None
        data = []
        for i in range(self.image.size[0]):
            for j in range(self.image.size[1]):
                data.append(self.pixels[i,j])
        self.X = np.array(data,dtype=np.float128)
        # normalize data from 0 to 255 to 0 to 1
        self.X /= 255
        self.dim = len(data[0])
        # init cluster num
        self.segment_num = segment_num
        # init pi vector with len segment_num
        self.pi_s = np.random.dirichlet(np.ones(self.segment_num))
        # init mu matrix with segment_num rows dim columns
        self.mu = np.random.uniform(0,1,(self.segment_num,self.dim))

    def e_step(self):
        # init soft ws, pixel_num rows segment_num columns
        self.w = np.zeros(shape=(self.pixel_num,self.segment_num))
        # calculate exp((xi - muj) T (xi - muj)) * pi_j
        self.x_mu = np.zeros(shape=(self.pixel_num,self.segment_num))
        for i in range(self.pixel_num):
            for j in range(self.segment_num):
                residual = self.X[i] - self.mu[j]
                self.x_mu[i][j] = np.exp(residual.dot(residual)*(-1/2)) * self.pi_s[j]
        for i in range(self.pixel_num):
            for j in range(self.segment_num):
                self.w[i][j] = self.x_mu[i][j] / np.sum(self.x_mu[i])
        # debug
        print 'E step done'

    def m_step(self):
        for j in range(self.segment_num):
            nom = np.zeros(self.dim)
            denom = 0.0
            for i in range(self.pixel_num):
                nom += self.X[i] * self.w[i][j]
                denom += self.w[i][j]
            # update muj
            self.mu[j] = nom / denom
            # update pij
            self.pi_s[j] = denom / self.pixel_num
        # debug
        print 'M step done'

    def likelihood(self):
        # calculate -1/2 (xi - muj) T (xi - muj) + log pi_j
        x_mu = np.zeros(shape=(self.pixel_num,self.segment_num))
        for i in range(self.pixel_num):
            for j in range(self.segment_num):
                residual = self.X[i] - self.mu[j]
                x_mu[i][j] = residual.dot(residual)*(-1/2) * np.log(self.pi_s[j])
        l = 0.0
        for i in range(self.pixel_num):
            for j in range(self.segment_num):
                l += x_mu[i][j] * self.w[i][j]
        return l

    def em(self):
        for i in range(1):
            self.e_step()
            self.m_step()
            l = self.likelihood()
            if self.l is not None:
                print 'relative difference in likelihood'
                relative = abs((l - self.l) / self.l)
                print relative
                if relative < 1e-6:
                    break
            self.l = l

    def nearest(self):
        result = np.zeros(self.pixel_num)
        for i in range(self.pixel_num):
            result[i] = self.w[i].argmax()
            print result[i]
        self.cluster = result
        return result

    def output(self):
        for i in range(self.width):
            for j in range(self.height):
                index = i * self.height + j
                pixel = self.mu[self.cluster[index]] * 255
                self.pixels[i,j] = tuple(pixel.astype(int))
                print self.pixels[i,j]
        plt.imshow(em.image)
        plt.show()

if __name__ == '__main__':
    em = GaussianEM("test_images/nature.jpg",10)
    em.em()
    em.nearest()
    em.output()