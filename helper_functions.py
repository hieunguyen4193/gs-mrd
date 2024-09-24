import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings("ignore")
import os
import ot
import pickle
import argparse
import random
##### Helper functions for OPTIMAL TRANSPORT DISTANCE

def calculate_barycenter(inputdf, samplelist, n, show_plot=False, M = None):
    first_sample = samplelist["Healthy"][0]

    A = inputdf[[first_sample]].to_numpy()
    for sampleid in samplelist["Healthy"][1:]:
        a2 = inputdf[[sampleid]].to_numpy()
        A = np.hstack((A, a2))
    n_distributions = A.shape[1]

    # loss matrix + normalization
    if M is None:
        M = ot.utils.dist0(n)
        M /= M.max()

    weights = [1/n_distributions for item in range(n_distributions)]
    x = np.arange(n, dtype=np.float64)
    # l2bary
    bary_l2 = A.dot(weights)

    if show_plot:
        f, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, num=1)
        ax1.plot(x, A, color="black")
        ax1.set_title('Distributions')

        ax2.plot(x, bary_l2, 'r', label='l2')
        ax2.set_title('Barycenters')

        plt.legend()
        plt.show()
    return bary_l2

def calculate_ot_distance_to_healthy_nuc(sample, bary_l2, inputdf, n):
    x = np.arange(n, dtype=np.float64)
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'euclidean')
    M /= M.max() * 0.1
    a = inputdf[sample].values
    a = np.array(a)
    b = bary_l2
    d_emd = ot.emd2(a, b, M)  # direct computation of OT loss
    return d_emd