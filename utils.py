import config as config
from feature_selection import *
from sklearn import metrics
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def regi_with_id(split,all_data,data_type,label):
    index = split['id'][split["group"] == data_type].values #[split[label] != 1]

    index_use=index.copy()
    for name in all_data.keys():
        index_use_new=[j for j in index_use if j in all_data[name].index]
        index_use=index_use_new

    index_use=sorted(index_use)
    feature_regi={}

    for name in all_data.keys():
        feature = all_data[name].loc[index_use]

        for c in feature.columns:
            if "label" in c or "LABEL" in c or "Label" in c:
                feature = feature.drop([c], axis=1)

        feature_regi[name]=feature

    label = pd.DataFrame(split.set_index("id").loc[index_use][label])
    # label[label!=1]=0
    leb = preprocessing.LabelEncoder()
    label_re = leb.fit_transform(label)
    label[label.columns[0]] = label_re
    return feature_regi,label


def data_split(all_data,split_csv,label,sheet_name=None):
    if not sheet_name:
        sheet_name = config.sheet_name[label]
    split = pd.read_excel(split_csv, sheet_name=sheet_name,dtype={'id': int, label: int, "group": str})
    val_feature, val_label = regi_with_id(split, all_data, "val",label)
    test_feature,test_label = regi_with_id(split, all_data, "test",label)
    train_feature,train_label = regi_with_id(split, all_data, "train",label)

    print(f"Train dataset has {len(train_label)},validation set has {len(val_label)},test set has{len(test_label)}")
    return train_feature,train_label,val_feature,val_label,test_feature,test_label


"""常用功能"""
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def HU_clip(img,HU):
    img[img>HU[1]]=HU[1]
    img[img<HU[0]]=HU[0]
    return img

def std_normlization(train_data,val_data="",test_data="",type="z-score"):
    if type=="z-score":
        std_m=preprocessing.StandardScaler()
    elif type=="min-max":
        std_m=preprocessing.MinMaxScaler()
    else:
        raise Exception("There is error in normalization")

    std_m.fit(train_data)
    train_scale=std_m.transform(train_data)

    if len(test_data):
        test_scale=std_m.transform(test_data)
    else:
        test_scale=[]

    if len(val_data):
        val_scale = std_m.transform(val_data)
    else:
        val_scale=[]

    return train_scale,val_scale,test_scale


def measure_matrix(x_test, y_test, clf="empty",y_pred_prob="empty"):
    if len(set(y_test))>2:
        if clf != "empty":
            y_pred_prob = clf.predict_proba(x_test)
        # y_pred = (np.argmax(y_pred_prob, 1) + 1).astype(int)
        y_pred = np.argmax(y_pred_prob, 1).astype(int)
        #AUC
        AUC = metrics.roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
        #spec,Sensitivity
        confusion = metrics.multilabel_confusion_matrix(y_test, y_pred)
        spec,Sensitivity=[],[]
        for con in confusion:
            tn,fp, fn, tp = con.ravel()
            spec .append(0 if tn + fp == 0 else tn / (tn + fp))
            Sensitivity .append(0 if tp + fn == 0 else tp / (tp + fn))
        spec=np.mean(spec)
        Sensitivity=np.mean(Sensitivity)
    else:
        if clf!="empty":
            y_pred_prob = clf.predict_proba(x_test)
        else:
            y_pred_prob=y_pred_prob
        y_pred_prob1=y_pred_prob[:,1]
        # y_pred_prob = clf.predict(x_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob1, pos_label=1)
        cutoff_ind = np.argmax(tpr - fpr)
        cutoff = thresholds[cutoff_ind]
        y_pred = (y_pred_prob1 >= cutoff).astype(int)
        AUC = metrics.roc_auc_score(y_test, y_pred_prob1)
        confusion = metrics.confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = confusion.ravel()
        spec = 0 if tn + fp == 0 else tn / (tn + fp)
        Sensitivity = 0 if tp + fn == 0 else tp / (tp + fn)

    acc = metrics.accuracy_score(y_test, y_pred)
    return AUC, spec, Sensitivity, acc,[y_pred_prob,y_pred,y_test,cutoff]

def verse_label_pred(label,pred):
    verse_label = label.copy()
    verse_label[label == 1] = 0
    verse_label[label == 0] = 1

    verse_pred = pred.copy()
    verse_pred[:,0] = pred[:,1]
    verse_pred[:,1] = pred[:,0]
    return verse_label,verse_pred

def measure_matrix_verse(x_test, y_test, clf="empty",y_pred_prob="empty"):
    if clf!="empty":
        y_pred_prob = clf.predict_proba(x_test)
    # y_pred_prob = clf.predict(x_test)[:, 1]
    y_test,y_pred_prob=verse_label_pred(y_test,y_pred_prob)
    y_pred_prob=y_pred_prob[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob, pos_label=1)
    AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    cutoff_ind = np.argmax(tpr - fpr)
    cutoff = thresholds[cutoff_ind]
    y_pred = (y_pred_prob >= cutoff).astype(int)
    acc = metrics.accuracy_score(y_test, y_pred)
    confusion = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion.ravel()
    spec = 0 if tn + fp == 0 else tn / (tn + fp)
    Sensitivity = 0 if tp + fn == 0 else tp / (tp + fn)

    return AUC, spec, Sensitivity, acc

"""Feature selection"""
def Selection_pipe(feature, label,method,n_components):
    step_method=method.split("+")
    train_data_fs=feature
    feature_list=[]
    for i in range(len(step_method)):
        m=step_method[i]
        print(m)

        if m in config.Method["Feature"]:
            train_data_fs, feature_index=feature_selection_with_Feature(train_data_fs,label,m)
        elif m in config.Method["Label"]:
            number=max(n_components,train_data_fs.shape[1]//2) if i<len(step_method)-1 else n_components #输入lasso的不止25个
            train_data_fs, feature_index,_ = feature_selection_with_Label(train_data_fs,label,m,n_feature=number)
        else:
            return f"The method chosen is not available: {m}"

        feature_list.append(len(feature_index))
        # print(feature_list)

        if i==0:
            feature_index_choose=feature_index
        else:
            feature_index_choose = feature_index_choose[feature_index]

    return feature_list,feature_index_choose


"""Classifier"""
def  Classification(model_name,train_data,val_data,test_data,params_search=config.params_search,only_get_model=False):
    #合并训练集和验证集
    ##采用smote解决样本不均衡问题 只在训练集上
    sm = SMOTE(random_state=666)
    train_data = sm.fit_resample(train_data[0],train_data[1])  # 即完成了合成
    # print(train_data[0].shape, train_data[1].shape)

    train_val_features = np.concatenate((train_data[0], val_data[0]), axis=0)
    train_val_labels = np.concatenate((train_data[1], val_data[1]), axis=0)
    val_fold = np.zeros(train_val_features.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    val_fold[:train_data[0].shape[0]] = -1
    ps = PredefinedSplit(test_fold=val_fold) #自定义验证集 cv=ps

    model = {
        'SVC': SVC,
        'LogR': LogisticRegression,
        'NaiveBayes': GaussianNB,
        "XGB": XGBClassifier,
        "LinR":LinearRegression,
    }

    param_space = {'estimator': model[model_name](),  # 目标分类器
                   'param_grid': params_search[model_name],  # 前面定义的我们想要优化的参数
                   'cv': ps,  # 交叉验证split策略
                   "scoring":'roc_auc',#多分类 ovr#roc_auc
                   "refit": False, #默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行
                   'n_jobs': -1,  # 并行运行的任务数，-1表示使用所有CPU
                   'verbose': 1,
                   }
    grsearch = GridSearchCV(**param_space)
    grsearch.fit(train_val_features, train_val_labels)  #feature label
    best_params=grsearch.best_params_
    # print("OVER GridSearchCV")

    clf=model[model_name](**best_params)
    if model_name=="XGB":
        eval_set=[(val_data[0],val_data[1])]
        clf.fit(train_data[0],train_data[1],eval_metric="logloss", eval_set=eval_set, verbose=True,early_stopping_rounds=10)

    else:
        clf.fit(train_data[0], train_data[1])

    final_model = clf
    if only_get_model:
        return final_model,best_params
    else:
        test_auc, test_spec, test_sens, test_acc,_ = measure_matrix(test_data[0], test_data[1],clf=final_model)
        val_auc, val_spec, val_sens, val_acc,_ = measure_matrix(val_data[0], val_data[1], clf=final_model)
        train_auc, train_spec, train_sens, train_acc,_ = measure_matrix(train_data[0], train_data[1], clf=final_model)
        print(f"TRAIN  AUV:{train_auc},SPEC:{train_spec},SENS:{train_sens},ACC:{train_acc}")
        print(f"VAL    AUV:{val_auc},SPEC:{val_spec},SENS:{val_sens},ACC:{val_acc}")
        print(f"TEST   AUV:{test_auc},SPEC:{test_spec},SENS:{test_sens},ACC:{test_acc}")
        return [train_auc, train_spec, train_sens, train_acc], [val_auc, val_spec, val_sens, val_acc],\
               [test_auc, test_spec, test_sens, test_acc],best_params



def preprocess_cli(df,task):
    df.replace({"Yes": 1,'YES': 1,'YES ': 1, "No": 0,"NO": 0,'N0':0, "yes": 1, "no": 0, "<0.9": 0.9, "<0.3": 0.3, "＜0.3": 0.3,
                "<2.0":2,"＜2":2,'＜2.0':2,"＜0.6":0.6,"<0.600":0.6,"<0.6":0.6,"<0.8":0.8,"/":0,"<1.73":1.73,">12000":12000}, inplace=True)
    if task=="T1" or task=="T3" or task=="Clinical":
        df.fillna(value=df.median(), inplace=True)
    else:
        df.dropna(axis=0, how='any', subset=["SCC","CA125","CA199","CA153"], inplace=True)
        df.fillna(value=df.median(), inplace=True)
    return df
