# -*- coding: utf-8 -*-
import numpy as np
import json
import re

"""
1 = train
2 = test
3 = dev
"""
TEST = "./data/test.json"
DEV = "./data/dev.json"
TRAIN = "./data/train.json"
DATA_FILE = [None, TRAIN, TEST, DEV]
DATA_JSON = [None, [], [], []]
LABEL = 5


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\-\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "-LRB-", string)
    string = re.sub(r"\)", "-RRB-", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def gat_label(sent):
    # [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    if sent >= 0 and sent <= .2:
        return 0
    elif sent <= .4:
        return 1
    elif sent <= .6:
        return 2
    elif sent <= .8:
        return 3
    elif sent <= 1.0:
        return 4
    else:
        return 2


def load_data_and_labels(data_file):
    # Load data from json file
    with open(data_file, "r", encoding='utf8') as f:
        items = json.load(f)
    text = [item[0] for item in items]
    label = []
    for item in items:
        la = [0] * LABEL
        la[item[1]] = 1
        label.append(la)
    # label = [item[1] for item in items]
    return (text, label)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            print ("start: %d, end: %d" % (start_index, end_index))
            yield shuffled_data[start_index:end_index]


def main():
    dictionary = {}
    with open("./stanfordSentimentTreebank/dictionary.txt", "r", encoding="utf8") as f:
        for line in f.readlines():
            key, value = line.strip().split("|")
            dictionary[clean_str(key)] = value


    split = {}
    with open("./stanfordSentimentTreebank/datasetSplit.txt", "r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            if i > 0:
                key, value = line.strip().split(",")
                split[key] = value


    sentiment = {}
    with open("./stanfordSentimentTreebank/sentiment_labels.txt", "r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            if i > 0:
                key, value = line.strip().split("|")
                sentiment[key] = value


    with open("./stanfordSentimentTreebank/datasetSentences.txt", "r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            if i > 0:
                index, sentence = line.strip().split("\t")
                # print (": ".join((index, sentence)))
                sentence = clean_str(sentence)
                DATA_JSON[int(split[index])].append((sentence, gat_label(float(sentiment[dictionary[sentence]]))))


    for i in range(1, 4):
        print ("json.dump %s" % i)
        with open(DATA_FILE[i], "w", encoding="utf8") as f:
            json.dump(DATA_JSON[i], f)


if __name__ == '__main__':
    main()
