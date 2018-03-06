import os
import re
import sklearn as sk
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

WORD_VECTOR_PATH = "data/wordVectors.txt"
VOCAB_PATH = "data/vocab.txt"
DATA_PATH = "data/primary_debates.csv"


def read_csv(in_file, lowercase=True):
    with open(in_file) as f:
        text, party = [], []
        next(f)
        reader = csv.reader(f)
        for line in reader:
            if len(line) == 7 and not line[1] == 'AUDIENCE' and not line[1] == 'OTHER':
                text.append(line[2].lower() if lowercase else line[2])
                if line[4] == 'Democratic':
                    party.append(0)
                else:
                    party.append(1)
    return text, party

def vector_map(word_vector_path, vocab_path):
    vocab = np.genfromtxt(vocab_path, dtype=str)
    vectors = np.genfromtxt(word_vector_path, dtype=str)
    for vec in vectors:
        vec = re.findall(r"[\w']+|[.,!?;]", "Hello, I'm a string!")
    vecmap = {}
    for i in range(len(vocab) - 1):
        vecmap[vocab[i]]= vectors[i]
    return vecmap

def embed_text(text, vecmap):
    text_arr = text.strip().split(' ')
    avg = np.sum(np.asarray([vecmap[key] for key in text_arr if key in vecmap], np.float32), axis=0)
    return avg / len(text_arr)

def set_params(net_size, function, update_rule, regularization_const, learning_rate_type, initial_learning_rate, batch, iter_bound):
    return MLPClassifier(hidden_layer_sizes = net_size, activation = function, solver = update_rule, alpha = regularization_const, batch_size = batch, learning_rate = learning_rate_type, learning_rate_init = initial_learning_rate, max_iter = iter_bound, shuffle = True)

def train_net(x_train, y_train):
    neural_net = set_params((100, 50,), 'logistic', 'sgd', 0.0001, 'adaptive', 0.001, 100, 1500)
    neural_net.fit(x_train, y_train)
    return neural_net

def main():
    vecmap = vector_map(WORD_VECTOR_PATH, VOCAB_PATH)
    raw_text, labels = read_csv(DATA_PATH)

    print float(sum(labels))/len(labels)

    text_vec = []
    for example in raw_text:
        text_vec.append(np.asarray(embed_text(example, vecmap)))

    new_text_vec = []
    new_labels = []
    for i in range(len(text_vec)):
        if text_vec[i].shape == (50,):
            new_text_vec.append(text_vec[i])
            new_labels.append(labels[i])

    text_vec = np.asarray(new_text_vec)
    labels = np.asarray(new_labels)

    x_train, x_test, y_train, y_test = train_test_split(text_vec, labels, test_size=.3, train_size=.7)
    print 'data loaded.'

    neural_net = train_net(x_train, y_train)
    print "The accuracy of the net on the validation set is " + str(float(np.sum(neural_net.predict(x_test) == y_test))/len(y_test))

if __name__ == "__main__":
    main()

