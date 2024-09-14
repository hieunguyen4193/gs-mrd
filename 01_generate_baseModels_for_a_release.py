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
from sklearn.mixture import GaussianMixture
import os
import ot
import pickle
import argparse
import Levenshtein
import itertools
from helper_functions import *

##### input args
PROJECT = "gs-mrd"
release_version = "06062024"
merge_version = "20240914"

##### configurations/paths
path_to_main_src = "/media/hieunguyen/HNSD01/src/gs-mrd"
path_to_merge_samples = f"{path_to_main_src}/all_samples/{merge_version}"
path_to_model_files = f"{path_to_main_src}/model_files/{release_version}"

##### metadata
metadata = pd.read_csv(f"{path_to_model_files}/release_metadata.csv")
general_metadata = pd.read_excel("All Samples GW_MRD_010924.modified.xlsx", index_col = [0])

metadata["Cancer"] = metadata["SampleID"].apply(lambda x: general_metadata[general_metadata["SampleID"] == x.split("-")[1]].Cancer.unique()[0])
metadata["True label"] = metadata["SampleID"].apply(lambda x: general_metadata[general_metadata["SampleID"] == x.split("-")[1]]["True label"].unique()[0])

##### feature matrix
motif_order = pd.read_csv("motif_order.csv").motif_order.to_list()

featuredf = dict()
for input_feature in ["EM", "FLEN", "NUCLEOSOME", "IchorCNA"]:
    tmpdf = pd.read_csv(f"{path_to_merge_samples}/{input_feature}_features.csv")
    tmpdf = tmpdf[tmpdf["SampleID"].isin(metadata.SampleID.unique())]
    if input_feature == "EM":
        featuredf[input_feature] = tmpdf[["SampleID"] + motif_order].copy()
    else:
        featuredf[input_feature] = tmpdf.copy()
        
    assert featuredf[input_feature].shape[0] == metadata.shape[0]

##### generate sample list for each class
samplelist = dict()
for label in metadata.Cancer.unique():
    samplelist[label] = metadata[metadata["Cancer"] == label]["SampleID"].to_list()

##### distance matrix based on edit distance of End motif 4bp
nucleotides = ['A', 'C', 'G', 'T']
motifs = [''.join(p) for p in itertools.product(nucleotides, repeat=4)]

# Initialize an empty distance matrix
distance_matrix = pd.DataFrame(index=motifs, columns=motifs)

# Compute the Levenshtein distance between each pair of 4-mer motifs
for motif1 in motifs:
    for motif2 in motifs:
        distance_matrix.loc[motif1, motif2] = Levenshtein.distance(motif1, motif2)

# Convert the distance matrix to integer type
M_EM = distance_matrix.to_numpy().copy()
M_EM /= M_EM.max() * 0.1

for input_feature in ["EM", "FLEN", "NUCLEOSOME"]:
    ##### generate average FEATURE in all control samples in this batch
    inputdf = featuredf[input_feature].copy().set_index("SampleID").T
    inputdf["Healthy"] = inputdf[samplelist["Healthy"]].mean(axis = 1)
    inputdf[["Healthy"]].to_csv(f"{path_to_model_files}/Healthy_reference_{input_feature}.csv")
    inputdf = inputdf.drop("Healthy", axis = 1)
    
    ##### calculate 
    if input_feature == "EM":
        baryl2 = calculate_barycenter(inputdf.to_numpy(), n = inputdf.shape[0], show_plot=False, M = M_EM)
    else: 
        baryl2 = calculate_barycenter(inputdf.to_numpy(), n = inputdf.shape[0], show_plot=False, M = None)
    pd.DataFrame(data = baryl2, columns = ["baryl2"]).to_csv(f"{path_to_model_files}/Healthy_OT_{input_feature}_baryl2.csv")
    
    ##### NMF models
    X = featuredf[input_feature].set_index("SampleID")
    model = NMF(n_components=2, init='random', random_state=0, solver = "mu")
    W = model.fit_transform(X.to_numpy())
    H = model.components_
    nmfdf = pd.DataFrame(data = W, columns = ["V1", "V2"])
    nmfdf["SampleID"] = list(X.index)
    nmfdf["V1_scale"] = nmfdf[["V1", "V2"]].apply(lambda x: x[0]/sum(x), axis = 1)
    nmfdf["V2_scale"] = nmfdf[["V1", "V2"]].apply(lambda x: x[1]/sum(x), axis = 1)
    nmfdf = nmfdf.merge(metadata, right_on = "SampleID", left_on = "SampleID")
    sns.lineplot(H[0, ], label = "Cancer")
    sns.lineplot(H[1, ], label = "Healthy")
    plt.legend()
    plt.show()

    signal1 = [i for i,j in enumerate(H[0, ]) if j == np.max(H[0, ])][0]
    signal2 = [i for i,j in enumerate(H[1, ]) if j == np.max(H[1, ])][0]

    if (signal1 < signal2):
        nmf_flen_cancer_signal = 1
    else:
        nmf_flen_cancer_signal = 2
    pd.DataFrame(data = [nmf_flen_cancer_signal], columns = ["nmf_flen_cancer_signal"]).to_csv(f"{path_to_model_files}/NMF_{input_feature}_cancer_signal.csv")
    filename = os.path.join(path_to_model_files, 'NMF_{input_feature}.sav')
    pickle.dump(model, open(filename, 'wb'))