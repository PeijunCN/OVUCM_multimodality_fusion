import os
import json
import numpy as np
import pandas as pd
import config
from utils import Selection_pipe
from multiprocessing import Pool

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError

SPLITS =1
FOLDs=5
CV_SPLITs=4
modality=["CT","DWI","T2","USC","USG"]
data = {}
task="3FIGO分期" # task name
data_splits=os.path.join(config.DATA_DIR, 'data_splits-{SPLITS}_fold5_cv4.json')
data_featrure=os.path.join(config.DATA_DIR,'data_features_splits-{SPLITS}_fold5_cv4.json')
method ="ANOVA+Spearman+Relief+Lasso"


def run(split):
    split=f"split_{split}"
    split_data=data[task][split]
    feature_task_split = {}

    for fold in list(split_data.keys()):
        if "fold" in fold:
            feature_task_split[fold]={}
            dev_dict=data[task][split][fold]["dev"]
            case=dev_dict["case"]
            label=dev_dict["label"]

            for m in modality:
                feature = feature_all_dict[m].loc[case]
                feature_name=feature.columns
                feature = np.asarray(feature)
                feature_list, feature_index = Selection_pipe(feature, label, method,n_components=60)
                feature_use=list(feature_name[feature_index])
                feature_task_split[fold][m]=feature_use
                print(f"{task}_{split}_{fold},{m}_features:{len(feature_use)}\n")
    return split,feature_task_split

with open(data_splits, 'r', encoding='utf-8') as f:
    data = json.load(f)

feature_all_dict = {}
for m in modality:
    feature_csv = pd.read_csv(os.path.join(config.DATA_DIR, f"Omics_Feats_{m}.csv"), index_col="index")
    feature_all_dict[m] = feature_csv

feature_dict = {}
feature_task = {}
splits=list(np.arange(0,SPLITS))
pool = Pool(20)
results = pool.map(run,splits)

for split,f_data in results:
    feature_task[split]=f_data

feature_dict[task] = feature_task
with open(data_featrure, 'w', encoding='utf-8') as f:
    json.dump(feature_dict, f, default=default, ensure_ascii=False)