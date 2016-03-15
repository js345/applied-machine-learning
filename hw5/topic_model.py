'''
cs498 topic_model
Created on 3/15/16
@author: xiaofo
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def logsumexp(logAi,logAmax):
    expsum = 0.0
    for l in range(len(logAi)):
        expsum += np.exp(logAi[l] - logAmax)
    return logAmax + np.log(expsum)

class TopicModelEM:

    def __init__(self, filename):
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

        self.doc_count = int(lines[0]) # N
        self.word_num = int(lines[1]) # M
        self.total_word_count = int(lines[2])
        self.topic_num = 30

        # reading input files
        # doc_count rows, word_num columns
        output = [[0 for j in range(self.word_num)] for i in range(self.doc_count)]

        for i in range(3,len(lines)):
            line = lines[i].split(' ')
            doc_id = int(line[0])
            word_id = int(line[1])
            word_count = int(line[2])
            output[doc_id-1][word_id-1] = word_count
        self.X = np.array(output, dtype=np.float128)
        # init pi vector of len topic_num
        self.pi_s = np.random.dirichlet(np.ones(self.topic_num))
        # init p_s matrix of topic_num rows and word_num columns
        self.p_s = np.random.dirichlet(np.ones(self.word_num),size=self.topic_num)
        # other useful vars
        self.l = None

    def e_step(self):
        # init soft ws, doc_count rows topic_num columns
        self.w = np.zeros(shape=(self.doc_count,self.topic_num))
        # calculate X (logP)^T and log pi
        self.log_p = np.log(self.p_s)
        self.log_p_T = np.transpose(self.log_p)
        self.R = self.X.dot(self.log_p_T)
        self.log_pi = np.log(self.pi_s)
        # calculate log Aij doc_count rows topic_num columns
        # log Aj = R[,j] + pi_j
        self.logA = np.zeros(shape=(self.doc_count,self.topic_num))
        self.logA = self.R + self.log_pi
        # log Amax for each row
        self.logAmax = np.amax(self.logA,axis=1)
        # calculate wij
        for i in range(self.doc_count):
            for j in range(self.topic_num):
                self.w[i][j] = np.exp(self.logA[i][j] - logsumexp(self.logA[i],self.logAmax[i]))
        # for debug & print
        print 'E step done'

    def m_step(self):
        # update ps
        for j in range(self.topic_num):
            nom = np.zeros(self.word_num)
            denom = 0.0
            wsum = 0.0
            for i in range(self.doc_count):
                nom += self.X[i] * self.w[i][j]
                denom += np.sum(self.X[i]) * self.w[i][j]
                wsum += self.w[i][j]
            self.p_s[j] = nom / denom
            # smoothing probability of words
            self.p_s[j] += 0.00001
            self.p_s[j] /= np.sum(self.p_s[j])
            # update pis
            self.pi_s[j] = wsum / self.doc_count
        # debugs
        print 'M step done'

    def likelihood(self):
        # calculate X (logP)^T and log pi
        self.log_p = np.log(self.p_s)
        self.log_p_T = np.transpose(self.log_p)
        self.R = self.X.dot(self.log_p_T)
        self.log_pi = np.log(self.pi_s)
        # calculate log Aij doc_count rows topic_num columns
        # log Aj = R[,j] + pi_j
        self.logA = np.zeros(shape=(self.doc_count,self.topic_num))
        self.logA = self.R + self.log_pi
        # log Amax for each row
        self.logAmax = np.amax(self.logA,axis=1)
        l = 0.0
        for i in range(self.doc_count):
            for j in range(self.topic_num):
                l += self.w[i][j] * self.logA[i][j]
        return l

    def em(self):
        for i in range(10):
            self.e_step()
            self.m_step()
            l = self.likelihood()
            if self.l is not None:
                print 'relative difference in likelihood'
                relative = abs((l - self.l) / self.l)
                print relative
                if relative < 1e-4:
                    break
            self.l = l

if __name__ == '__main__':
    em = TopicModelEM('docword.nips.txt')
    em.em()
    plt.plot(em.pi_s)
    plt.title('Probability with which the topic is selected')
    plt.show()
    vocabs = np.loadtxt('vocab.nips.txt',dtype=str)
    for i in range(30):
        sorted_index = np.argsort(em.p_s[i])
        print "topic " + str(i)
        words = ""
        ps = ""
        for index in sorted_index[-10:]:
            print vocabs[index] + " : " + str(em.p_s[i][index]) + ", ",
        print "\n"
