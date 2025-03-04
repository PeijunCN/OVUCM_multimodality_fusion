# -*- coding: utf-8 -*-
import argparse
from utils import *
import config
import pickle
import json
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError

def middle_fusion(train_feature):
    for i,name in enumerate(train_feature.keys()):
        if not i:
            feature=train_feature[name]
        else:
            feature=np.concatenate((feature,train_feature[name]), axis=1)

    return feature


if __name__=="__main__":
    parser = argparse.ArgumentParser('set some args in the command line')
    parser.add_argument('-task', '--task', required=False, type=str, default='1_1良非良')
    parser.add_argument('-model', '--model', required=False, type=str, default="SVC")
    parser.add_argument('-combin', '--combin', required=False, type=str, default="CT")
    parser.add_argument('-num', '--num', required=False, type=int, default=40)
    args = parser.parse_args()

    task=args.task
    model_name=args.model


    combin = args.combin
    modality = sorted(combin.split("_"))

    n_components_list=[args.num]

    """paths"""
    ALL_PATH = config.DATA_DIR
    params_search = config.params_search
    data_splits_path=os.path.join(config.DATA_DIR, 'data_splits-1_fold5_cv4.json')
    data_features_path=os.path.join(config.DATA_DIR, 'data_features_splits-1_fold5_cv4.json')
    if not n_components_list[0]:
        n_components_list = np.arange(10, 51, 10).tolist()
        n_components_list += [5, 15]

    with open(data_splits_path, 'r', encoding='utf-8') as f:
        data_splits = json.load(f)

    with open(data_features_path, 'r', encoding='utf-8') as f:
        features = json.load(f)

    data_names = {}
    data_prep = {}

    task_data=data_splits[task][f"split_0"]
    feature_task={}
    save_path = os.path.join(config.Program_Path,task,f"split-0","_".join(modality),model_name)
    os.makedirs(save_path, exist_ok=True)

    modality_use = modality + ["Clinical"]
    for Total_Num in n_components_list:

        prob_dict = {}
        best_params_dict = {}
        n_list=[Total_Num//len(modality),]*len(modality)
        for i in range(Total_Num%len(modality)):
            n_list[i]+=1

        # print(modality_use)
        for fold in list(task_data.keys()):
            if "fold" not in fold:
                continue
            feature_name=features[task][f"split_0"][fold]
            feature_dict={}
            best_params_dict[fold]={}
            prob_dict[fold] = {}

            fold_dev_data_modal = {}
            fold_test_data_modal = {}

            test_case = task_data[fold]["test"]["case"]
            test_label = task_data[fold]["test"]["label"]

            dev_case = task_data[fold]["dev"]["case"]
            dev_label = task_data[fold]["dev"]["label"]

            val_auc_best = 0
            for index,cv in enumerate(list(task_data[fold]["dev_cv"].keys())):
                print(f"feature_{Total_Num}-{fold}-{cv}")
                train_case = task_data[fold]["dev_cv"][cv]["train"]["case"]
                train_label = task_data[fold]["dev_cv"][cv]["train"]["label"]

                val_case = task_data[fold]["dev_cv"][cv]["val"]["case"]
                val_label = task_data[fold]["dev_cv"][cv]["val"]["label"]

                train_data_modal={}
                val_data_modal={}
                test_data_modal={}

                for n_index,m in enumerate(modality_use):
                    if m=="Clinical":
                        train_data = np.asarray(task_data[fold]["dev_cv"][cv]["train"]["cli"])[:,:11].astype(np.float64)
                        val_data = np.asarray(task_data[fold]["dev_cv"][cv]["val"]["cli"])[:,:11].astype(np.float64)
                        test_data=np.asarray(task_data[fold]["test"]["cli"])[:,:11].astype(np.float64)
                        train_data_nor, val_data_nor, test_data_nor = std_normlization(train_data, val_data, test_data)

                        if not index:
                            dev_data = np.asarray(task_data[fold]["dev"]["cli"])[:,:11].astype(np.float64)
                            fold_dev_data_nor,_, fold_test_data_nor = std_normlization(dev_data,[], test_data)

                    else:
                        n=n_list[n_index]
                        print(m,n)
                        feature_csv=pd.read_csv(os.path.join(config.DATA_DIR,f"Omics_Feats_{m}.csv"),index_col="index")
                        train_data = np.asarray(feature_csv.loc[train_case,feature_name[m]])[:,:n]
                        val_data = np.asarray(feature_csv.loc[val_case, feature_name[m]])[:,:n]
                        test_data = np.asarray(feature_csv.loc[test_case, feature_name[m]])[:, :n]
                        train_data_nor, val_data_nor, test_data_nor = std_normlization(train_data, val_data, test_data)

                        if not index:
                            dev_data = np.asarray(feature_csv.loc[dev_case, feature_name[m]])[:, :n]
                            fold_dev_data_nor,_, fold_test_data_nor = std_normlization(dev_data,[], test_data)



                    test_data_modal[m] = test_data_nor
                    train_data_modal[m] = train_data_nor
                    val_data_modal[m] = val_data_nor

                    if not index:
                        fold_dev_data_modal[m]=fold_dev_data_nor
                        fold_test_data_modal[m]=fold_test_data_nor

                train_data_use = [middle_fusion(train_data_modal), train_label]
                test_data_use = [middle_fusion(test_data_modal), test_label]
                val_data_use = [middle_fusion(val_data_modal), val_label]
                # print(test_data_use[0].shape)
                final_model,best_params = Classification(model_name, train_data_use,val_data_use, test_data_use,params_search,only_get_model=True)
                best_params_dict[fold][f"{cv}-params"] = best_params
                if model_name != "LinR":
                    prob_dict[fold][f"{cv}-train"] = final_model.predict_proba(train_data_use[0])[:,1]
                    prob_dict[fold][f"{cv}-val"] = final_model.predict_proba(val_data_use[0])[:,1]
                    prob_dict[fold][f"{cv}-test"] = final_model.predict_proba(test_data_use[0])[:,1]
                else:
                    prob_dict[fold][f"{cv}-train"] = final_model.predict(train_data_use[0])
                    prob_dict[fold][f"{cv}-val"] = final_model.predict(val_data_use[0])
                    prob_dict[fold][f"{cv}-test"] = final_model.predict(test_data_use[0])

                val_AUC = metrics.roc_auc_score(val_data_use[1], prob_dict[fold][f"{cv}-val"])
                if val_AUC>val_auc_best:
                    val_auc_best=val_AUC
                    best_model=final_model
                    best_fold_params=best_params

                save_path_feature=os.path.join(save_path,f"feature_{Total_Num}")
                os.makedirs(save_path_feature,exist_ok=True)
                with open(os.path.join(save_path_feature,f"feature_{Total_Num}-{fold}-{cv}.pkl"), 'wb') as f:
                    pickle.dump(final_model, f)

            dev_data_use = [middle_fusion(fold_dev_data_modal), dev_label]
            test_data_use = [middle_fusion(fold_test_data_modal), test_label]
            sm = SMOTE(random_state=666)
            dev_data = sm.fit_resample(dev_data_use[0], dev_data_use[1])
            best_model.fit(dev_data[0], dev_data[1])
            fold_model = best_model
            best_params_dict[fold]["best_params"]=best_fold_params

            if model_name!="LinR":
                prob_dict[fold]["dev"] = fold_model.predict_proba(dev_data_use[0])[:,1]
                prob_dict[fold]["test"] = fold_model.predict_proba(test_data_use[0])[:,1]
            else:
                prob_dict[fold]["dev"] = fold_model.predict(dev_data_use[0])
                prob_dict[fold]["test"] = fold_model.predict(test_data_use[0])

            save_path_best_model = os.path.join(save_path, "best_model_of_fold")
            os.makedirs(save_path_best_model, exist_ok=True)
            with open(os.path.join(save_path_best_model, f"best_model-feature_{Total_Num}-{fold}.pkl"), 'wb') as f:
                pickle.dump(fold_model, f)


        with open(os.path.join(save_path, f"feature_{Total_Num}-prob_results.json"), 'w', encoding='utf-8') as f:
            json.dump(prob_dict, f, default=default, ensure_ascii=False)

        with open(os.path.join(save_path, f"feature_{Total_Num}-params.json"), 'w', encoding='utf-8') as f:
            json.dump(best_params_dict, f, default=default, ensure_ascii=False)


