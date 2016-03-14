from __future__ import division
import numpy as np

class EM:
    def __init__(self, file_name): #'./docword.nips.txt'
        f = open(file_name, 'r')
        lines = f.readlines()
        f.close()

        self.doc_count = int(lines[0]) # N
        self.word_num = int(lines[1])
        self.total_word_count = int(lines[2])
        self.topic_num = 30

        output = [[0 for j in range(self.word_num)] for i in range(self.doc_count)]

        i = 3
        while i < len(lines):
            line = lines[i].split(' ')
            doc_id = int(line[0])
            word_id = int(line[1])
            word_count = int(line[2])
            output[doc_id-1][word_id-1] = word_count
            i += 1

        #print(output[0])

        self.data = np.array(output, dtype=np.float128)

        self.pi_s = np.random.rand(self.topic_num) #np.array([1/30 for i in range(self.topic_num)], dtype=np.float128)
        self.pi_s = self.pi_s /  np.sum(self.pi_s)
        # self.pi_s = np.zeros(30)
        # self.pi_s[0] = 1

        self.p_s = np.array([np.random.rand(self.word_num) for j in range(30)], dtype=np.float128)
        for i in range(30):
            sum_ = np.sum(self.p_s[i])
            for j in range(self.word_num):
                self.p_s[i][j] /= sum_

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

        R = self.data.dot(np.log10(self.p_s).T)
        log_pi = np.log10(self.pi_s)

        for i in range(R.shape[0]):
            sum_ = 0.0
            max_ = 0.0
            for j in range(R.shape[1]):
                w[i][j] = R[i, j] + log_pi[j]
                print R[i,j] + log_pi[j]

                if (w[i][j] < max_):
                    max_ = w[i][j]
            for j in range(R.shape[1]):
                w[i][j] -= max_
                sum_ += w[i][j]
            for j in range(R.shape[1]):
                w[i][j] /= (sum_)
        """for i in range(R.shape[0]):
            sum_ = 0
            #log_Amax = -999999.0
            for j in range(R.shape[1]):
                w[i, j] = 10 ** (R[i, j] + log_pi[j])

                #if w[i, j] > log_Amax:
                #    log_Amax = w[i, j]

                sum_ += w[i, j]#10 ** (w[i, j] - log_Amax)

            for j in range(R.shape[1]):
                w[i, j] = 10** np.log10(w[i, j]) - np.log10(sum_)#w[i, j] = w[i, j] - log_Amax - np.log10(sum_)
                #w[i, j] = 10 ** w[i, j]"""

        print(np.sum(w, axis = 1))
        print(np.sum(w.T, axis = 1))

        self.w = w
        print w
        print('done e_step')

    def m_step(self):
        for j in range(self.topic_num):
            numer = 0
            denom = 0

            #self.pi_s[j] = 0
            for i in range(self.doc_count):
                numer += (self.data[i] * self.w[i][j])
                denom += (np.sum(self.data[i]) * self.w[i][j])

                # self.pi_s[j] += self.w[i, j]

            self.p_s[j] = numer / denom

            # self.pi_s[j] = self.pi_s[j] / self.doc_count
            self.pi_s[j] = (np.sum(self.w[:, j])) / self.doc_count

        print('done m_step')
        print(np.sum(self.pi_s))
        print(np.sum(self.p_s[0]))

    def em_step(self):
        self.e_step()
        self.m_step()

        print(self.p_s)
        print(self.pi_s)

        # smooth
        self.p_s += 0.02
        for i in range(self.p_s.shape[0]):
            self.p_s[i] = self.p_s[i] / np.sum(self.p_s[i])

        self.e_step()
        self.m_step()

        print(self.p_s)
        print(self.pi_s)


if __name__ == '__main__':
    em = EM('docword.nips.txt')
    em.em_step()
