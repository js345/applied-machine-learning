from __future__ import division
from PIL import Image
import numpy as np

class EM:
    def __init__(self, file_name):
        self.image = Image.open(file_name)
        self.pixels = self.image.load()
        self.width = self.image.size[0]
        self.height = self.image.size[1]

        data = []
        for i in range(self.image.size[0]):
            for j in range(self.image.size[1]):
                data.append(self.pixels[i, j])

        self.data = np.array(data, dtype=np.float128) # M
        self.N = len(self.data)

        self.k = 10  # num of clusters

        self.pi_s = np.random.rand(self.k)
        self.pi_s = self.pi_s /  np.sum(self.pi_s)

        r_mean = np.mean(self.data[:, 0])
        g_mean = np.mean(self.data[:, 1])
        b_mean = np.mean(self.data[:, 2])
        self.mu = np.array([np.random.randint(255, size=3) for j in range(self.k)], dtype=np.float128)  # k x 3

        print(self.mu)
        print(self.pi_s)

    def e_step(self):
        w = np.array([np.zeros(self.k) for i in range(self.N)], dtype=np.float128)

        '''
        for i in range(self.N):
            sum_ = 0
            for j in range(self.k):
                w[i, j] = np.exp(-1/2 * (self.data[i] - self.mu[j]).dot(self.data[i] - self.mu[j])) * self.pi_s[j]

                sum_ += w[i, j]
            # print(sum_)
            w[i, :] /= sum_
        self.w = w
        '''

        for i in range(self.N):
            A_max = None
            for j in range(self.k):
                A_ij = -1/2 * (self.data[i] - self.mu[j]).dot(self.data[i] - self.mu[j])
                w[i, j] = A_ij

                if A_max == None or A_ij > A_max:
                    A_max = A_ij

            # calculate sum_
            sum_ = 0
            for j in range(self.k):
                A_ij = w[i, j]
                sum_ += np.exp(A_ij - A_max) * self.pi_s[j]

            for j in range(self.k):
                A_ij = w[i, j]
                log_w_ij = A_ij + np.log(self.pi_s[j]) - A_max - np.log(sum_)
                w[i, j] = np.exp(log_w_ij)

        self.w = w

    def m_step(self):
        for j in range(self.k):
            numer = 0
            denom = 0
            for i in range(self.N):
                numer += self.data[i] * self.w[i, j]
                denom += self.w[i, j]
            self.mu[j] = numer / denom
            self.pi_s[j] = denom / self.N

    def em_step(self):
        self.e_step()
        self.m_step()

        self.pi_s += 0.0001
        self.pi_s /= sum(self.pi_s)

        print self.mu
        print self.pi_s
        print np.sum(self.pi_s)


        self.e_step()
        self.m_step()

        print self.mu
        print self.pi_s
        print np.sum(self.pi_s)

if __name__ == '__main__':
    em = EM("test_images/nature.jpg")
    em.em_step()
