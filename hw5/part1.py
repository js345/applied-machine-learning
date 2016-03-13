import numpy as np


class EM:
    def __init__(self, file_name): #'./docword.nips.txt'
        f = open(file_name, 'r')
        lines = f.readlines()

        self.doc_count = int(lines[0])
        self.word_num = int(lines[1])
        self.total_word_count = int(lines[2])

        i = 3
        output = [[0]*word_num for i in range(doc_count)]
        while i < len(lines):
            line = lines[i].strip().split(' ')
            doc_id = int(line[0])
            word_id = int(line[1])
            word_count = int(line[2])
            output[doc_id-1][word_id-1] = word_count
            i += 1
            
        self.data = np.array(output)


if __name__ == '__main__':
    # print load_data()
