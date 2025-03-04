import os
import json
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import StratifiedKFold
from utils import preprocess_cli
import random

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError

label_path=config.clinical_and_label_path
task_label_dict=config.sheet_name

SPLITS=1
FOLDs=5
CV_SPLITs=4
skf_cv = StratifiedKFold(n_splits=CV_SPLITs,shuffle=True)
skf_split = StratifiedKFold(n_splits=FOLDs,shuffle=True)

data_dict={}
# 1.read data by task
for index,sheet_name in enumerate(task_label_dict.keys()):
    print(sheet_name)
    data_dict[sheet_name]={}


    cli_data = config.cli_withca

    label=task_label_dict[sheet_name]
    data=pd.read_excel(label_path, sheet_name=sheet_name,dtype={'id': int, label: float})
    data = data.drop(np.where(np.isnan(data["Fold"]))[0])

    data_x = np.asarray(data["id"])
    data_y = np.asarray(data[label])

    data=data.set_index("id")

    # split:50
    for split in range(SPLITS):
        data_xy = list(zip(data_x,data_y))
        random.shuffle(data_xy)
        data_x[:], data_y[:] = zip(*data_xy)

        # fold:5
        task_dict = {}
        for fold,(dev_index, test_index) in enumerate(skf_split.split(data_x, data_y)):
            task_dict["clinic"] = cli_data
            fold_name=f"fold_{fold}"
            task_dict[fold_name]={"test":{},"dev":{}}

            test_x, test_y = data_x[test_index], data_y[test_index]
            test_cli = data.loc[test_x,cli_data]
            task_dict[fold_name]["test"]={"case":test_x,"label":test_y,"cli":test_cli}

            dev_x, dev_y = data_x[dev_index], data_y[dev_index]
            dev_cli = data.loc[dev_x,cli_data]
            task_dict[fold_name]["dev"] = {"case": dev_x, "label": dev_y,"cli":dev_cli}

            # cv:4
            fold_cv_dict={}
            for cv,(train_index,val_index) in enumerate(skf_cv.split(dev_x, dev_y)):
                cv_name=f"cv_{cv}"
                fold_cv_dict[cv_name]={"train":{},"val":{}}

                train_x,train_y=dev_x[train_index],dev_y[train_index]
                train_cli = data.loc[train_x,cli_data]

                val_x, val_y = dev_x[val_index], dev_y[val_index]
                val_cli = data.loc[val_x,cli_data]


                fold_cv_dict[cv_name]["train"] = {"case": train_x, "label": train_y,"cli":train_cli}
                fold_cv_dict[cv_name]["val"] = {"case": val_x, "label": val_y,"cli":val_cli}
                print(f"SPLIT{split},FOLD{fold},CV{cv},train_cases:{len(train_y)},val_cases:{len(val_y)}")

            task_dict[fold_name]["dev_cv"]=fold_cv_dict
            print(f"SPLIT{split},FOLD{fold},dev_cases:{len(dev_y)},test_cases:{len(test_y)}\n")

        data_dict[sheet_name][f"split_{split}"]=task_dict



# 存入json
with open(os.path.join(config.DATA_DIR, f'data_splits-{SPLITS}_fold5_cv4.json'), 'w', encoding='utf-8') as f:
    json.dump(data_dict,f,default=default,ensure_ascii=False)


# 读取 JSON 文件
with open(os.path.join(config.DATA_DIR, f'data_splits-{SPLITS}_fold5_cv4'), 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data)







