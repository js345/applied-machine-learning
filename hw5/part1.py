import numpy as np

def load_data():
    f = open('./docword.nips.txt', 'r')
    lines = f.readlines()

    doc_count = int(lines[0])
    word_num = int(lines[1])
    total_word_count = int(lines[2])

    i = 3
    output = [[0]*word_num for i in range(doc_count)]
    while i < len(lines):
        line = lines[i].strip().split(' ')
        doc_id = int(line[0])
        word_id = int(line[1])
        word_count = int(line[2])
        output[doc_id-1][word_id-1] = word_count
        i += 1
    return np.array(output)

if __name__ == '__main__':
    print load_data()
