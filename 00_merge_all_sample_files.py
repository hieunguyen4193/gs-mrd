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

PROJECT = "gs-mrd"
data_merge_version = "20240914"
maindir = "/media/hieunguyen/GSHD_HN01"
path_to_storage = os.path.join(maindir, "storage")
path_to_main_src = "/media/hieunguyen/HNSD01/src/gs-mrd"
path_to_save_merge_feature = os.path.join(path_to_main_src, "all_samples", data_merge_version)
os.system(f"mkdir -p {path_to_save_merge_feature}")

##### modify metadata
if os.path.isfile("All Samples GW_MRD_010924.modified.xlsx") == False:
    metadata = pd.read_excel("All Samples GW_MRD_010924.xlsx")
    '''
    These labcodes have been changed and need to be converted in the metadata.
    '''
    convert_labcode = {
        "HMAAAA03": "ZTKL01A",
        "HMAAAA26": "ZTKL05A",
        "HMAAAA21": "ZTKL07A",
        "ZMC031A": "ZMC031",
        "ZMC057A": "ZMC057",
        "ZMC005A": "ZMC005",
        "ZMG093A": "ZMG093",
        "MDCAAA03": "MQCAAA03",
        "MDAAAA18": "MQAAAA18",
        "ZMG040A": "ZMC040A"
    }
    metadata["SampleID"] = metadata["SampleID"].apply(lambda x: convert_labcode[x] if x in convert_labcode.keys() else x)
    metadata = metadata[~metadata["SampleID"].duplicated()]
    metadata.to_excel("All Samples GW_MRD_010924.modified.xlsx")
else:
    metadata = pd.read_excel("All Samples GW_MRD_010924.modified.xlsx")

rerun_samples = pd.read_csv("rerun_samples_not_used.txt", header = None)[0].unique()

featuredf = dict()
all_files = dict()
for input_feature in ["EM", "FLEN", "NUCLEOSOME", "IchorCNA"]:
    all_files[input_feature] = [item for item in pathlib.Path(os.path.join(path_to_storage, PROJECT)).glob("*/*/*{}*.csv".format(input_feature))]
    print(f"Found {len(all_files[input_feature])} files in {input_feature} feature")
        
    featuredf[input_feature] = pd.DataFrame()
    batch_metadata = pd.DataFrame()
    
    for file in all_files[input_feature]:
        batch_name = str(file).split("/")[-3]
        run_name = str(file).split("/")[-2]
        tmpdf = pd.read_csv(file, index_col = [0])
        tmpdf.columns = ["SampleID"] + list(tmpdf.columns)[1:]
        tmpdf["SampleID"] = tmpdf["SampleID"].apply(lambda x: x.split("_")[0])
        tmp_batch_metadata = pd.DataFrame(
            {
                "SampleID": tmpdf.SampleID.unique(),    
                "RUN": [run_name for item in tmpdf.SampleID.unique()],
                "Group_RUN": [batch_name for item in tmpdf.SampleID.unique()]
            }
        )
        batch_metadata = pd.concat([batch_metadata, tmp_batch_metadata], axis = 0)
        featuredf[input_feature] = pd.concat([featuredf[input_feature], tmpdf], axis = 0)

    batch_metadata.to_csv(os.path.join(path_to_save_merge_feature, f"{input_feature}_batch_metadata.csv"), index = False)

    featuredf[input_feature] = featuredf[input_feature][~featuredf[input_feature]["SampleID"].isin(rerun_samples)]
    
    # featuredf[input_feature]["SampleID"] = featuredf[input_feature]["SampleID"].apply(lambda x: x.split("-")[1])
    
    ##### remove duplicated labcodes
    featuredf[input_feature] = featuredf[input_feature][~featuredf[input_feature].duplicated()]
    
    ##### temporarily remove samples that are not in the metadata
    # featuredf[input_feature] = featuredf[input_feature][featuredf[input_feature]["SampleID"].isin(metadata["SampleID"].unique())]
    print(f"There are {featuredf[input_feature].shape[0]} samples in {input_feature} feature")
    print(f"There are {featuredf[input_feature].shape[1]} feature in {input_feature} feature")

    ##### save feature matrix to file
    if os.path.isfile(os.path.join(path_to_save_merge_feature, f"{input_feature}_features.csv")) == False:
        featuredf[input_feature].to_csv(os.path.join(path_to_save_merge_feature, f"{input_feature}_features.csv"), index = False)
    else:
        print(f"File exists, do not generate new data")    