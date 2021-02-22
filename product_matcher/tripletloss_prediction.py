import numpy as np
import os
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.spatial.distance import cdist
import cv2 as cv
import tensorflow as tf
from tripletloss_preprocessing import PreProcessing
from tripletloss_model import TripletLoss


# helper function to plot image
def show_image(idxs, data):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs]) # 2d array
    fig = plt.figure()
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)): # iterate through image indexes
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(data[idxs[i],:,:,:])
        ax.axis('off')
    plt.show()
    