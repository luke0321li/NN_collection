import string
import unicodedata
import numpy as np


def generate_data(input_file):
    data = []
    word2num = {}
    num2word = {}
    num = 0
    table = str.maketrans({key: None for key in string.punctuation})
    with open(input_file) as file:
        for line in file:
            text = line.translate(table).split()
            for word in text:
                word = word.lower()
                if word not in word2num:
                    word2num[word] = num
                    num += 1
                data.append(word2num[word])

    for key, value in word2num.items():
        num2word[value] = key

    return data, word2num, num2word


word_index = 0


def get_batch(data, batch_size, half_context_len):
    global word_index
    contexts = []
    labels = []
    for i in range(batch_size):
        if word_index + half_context_len * 2 + 1 > len(data):
            word_index = 0
        labels.append(data[word_index + half_context_len])
        contexts.append([data[j] for j in range(word_index, word_index + half_context_len * 2 + 1) if
                         j != word_index + half_context_len])

        word_index += 1
    return contexts, labels


# data, word2num, num2word = generate_data("test.txt")
# contexts, labels = get_batch(data, 10, 2)
