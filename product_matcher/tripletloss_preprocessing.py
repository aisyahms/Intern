# adapted from https://github.com/sanku-lib/image_triplet_loss/blob/master/preprocessing.py

import os
from matplotlib.image import imread
import numpy as np
import pandas as pd

class PreProcessing:

    images_train = np.array([])
    images_valid = np.array([])
    labels_train = np.array([])
    labels_valid = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self,data_src):
        self.data_src = 'retail_corpus/'
        print("Loading Retail Corpus Dataset...")
        self.images_train, self.images_valid, self.labels_train, self.labels_valid = self.preprocessing(0.9) # 90% training 
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in
                                        self.unique_train_label}
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Images validation :", self.images_valid.shape)
        print("Labels validation  :", self.labels_valid.shape)
        print("Unique label :", self.unique_train_label)

    def normalize(self,x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_dataset(self):
        X = []
        y = []
        for directory in os.listdir(self.data_src):
            try:
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = imread(os.path.join(self.data_src, directory, pic))
                    X.append(np.squeeze(np.asarray(img)))
                    y.append(directory)
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X,y

    def preprocessing(self,train_valid_ratio):
        X, y = self.read_dataset()
        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y]) # index
        X = [self.normalize(x) for x in X]                                  # normalize images

        shuffle_indices = np.random.permutation(np.arange(len(y)))          # index randomised to random split between train and test
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])

        size_of_dataset = len(x_shuffled)
        n_train = int(np.ceil(size_of_dataset * train_valid_ratio))
        return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(y_shuffled[0:n_train]), np.asarray(y_shuffled[n_train + 1:size_of_dataset])

    def get_triplets(self):
        label_l, label_r = np.random.choice(self.unique_train_label, 2, replace=False) # label for similar, label for dissimilar
        a, p = np.random.choice(self.map_train_label_indices[label_l],2, replace=False) # anchor, positive image (similar)
        n = np.random.choice(self.map_train_label_indices[label_r]) # negative image
        return a, p, n

    def get_triplets_batch(self,n):
        idxs_a, idxs_p, idxs_n = [], [], []
        for _ in range(n):
            a, p, n = self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
        return self.images_train[idxs_a,:], self.images_train[idxs_p, :], self.images_train[idxs_n, :]
    