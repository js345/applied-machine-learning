from __future__ import division
import numpy as np

class EM:
    def __init__(self, file_name): #'./docword.nips.txt'
        f = open(file_name, 'r')
        lines = f.readlines()

        self.doc_count = int(lines[0]) # N
        self.word_num = int(lines[1])
        self.total_word_count = int(lines[2])
        self.topic_num = 30

        i = 3
        output = [[0]*self.word_num for i in range(self.doc_count)]
        while i < len(lines):
            line = lines[i].strip().split(' ')
            doc_id = int(line[0])
            word_id = int(line[1])
            word_count = int(line[2])
            output[doc_id-1][word_id-1] = word_count
            i += 1

        self.data = np.array(output, dtype=np.float128)

        self.pi_s = np.array([1/30 for i in range(self.topic_num)], dtype=np.float128)
        self.p_s = np.array([[1/self.word_num for i in range(self.word_num)] for j in range(30)], dtype=np.float128)

    def e_step(self):
        # i = range(self.doc_count)
        # j = range(30)
        w = np.array([[0 for j in range(self.topic_num)] for i in range(self.doc_count)], dtype=np.float128)

        '''
        def product(end, i, j):
            output = np.float128(1)
            for k in range(end):
                output *= (self.p_s[j][k] ** self.data[i][k])

                if self.p_s[j][k] == 0:
                    print('fcuk 0')
                    exit(0)
            return output

        for i in range(self.doc_count):
            sum_ = 0
            for j in range(self.topic_num):
                w[i, j] = product(self.word_num, i, j) * self.pi_s[j]
                print('@ ', product(self.word_num, i, j), self.pi_s[j])
                print('here ', w[i, j])
                sum_ += w[i, j]
            # sum_ = sum(w[i])

            print(sum_)

            for j in range(self.topic_num):
                w[i, j] /= sum_
            #w[i] /= sum_
        '''

        R = self.data.dot(np.log(self.p_s).T)
        log_pi = np.log(self.pi_s)

        for i in range(R.shape[0]):
            sum_ = 0
            log_Amax = 0
            for j in range(R.shape[1]):
                w[i, j] = np.e ** (R[i, j] + log_pi[j])

                if w[i, j] > log_Amax:
                    log_Amax = w[i, j]

                sum_ += np.e ** (w[i, j] - log_Amax)

            for j in range(R.shape[1]):
                w[i, j] = w[i, j] - log_Amax - np.log(sum_)
                w[i, j] = np.e ** w[i, j]
        # print(w)

        self.w = w

        print('done e_step')

    def m_step(self):
        for j in range(self.topic_num):
            numer = 0
            denom = 0

            self.pi_s[j] = 0
            for i in range(self.doc_count):
                numer += (self.data[i] * self.w[i][j])
                denom += (sum(self.data[i]) * self.w[i][j])

                self.pi_s[j] += self.w[i][j]

            self.p_s[j] = numer / denom
            self.pi_s[j] = self.pi_s[j] / self.doc_count
            # self.pi_s[j] = (np.sum(self.w[:, j])) / self.doc_count

        print('done m_step')
        print(np.sum(self.pi_s))

    def em_step(self):
        self.e_step()
        self.m_step()

        print(self.pi_s)

if __name__ == '__main__':
    em = EM('docword.nips.txt')
    em.em_step()

