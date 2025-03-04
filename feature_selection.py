import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn import preprocessing
from sklearn.linear_model import LassoCV, LogisticRegression
from scipy import stats
from scipy.stats import ttest_ind, levene
from skfeature.function.information_theoretical_based import MIM, DISR
from skrebate import ReliefF
from skfeature.function.similarity_based import fisher_score
import mifs
'''
特征间相关的特征筛选，ind为所选的特征索引
Pearson相关性 Spearman相关性 Mann-Whitney U_test T-test
零假设的非参数检验，即样本 x 的分布与样本 y 的分布
Pearson相关系数衡量两个数据集之间的线性关系，这个系数在 -1 和 +1 之间变化，0 表示没有相关性
Spearman相关系数衡量两个数据集之间的线性关系，这个系数在 -1 和 +1 之间变化，0 表示没有相关性
U_test  遇见全0项会报错　ValueError: All numbers are identical in mannwhitneyu
'''
def feature_selection_with_Feature(X, y, method, *args):
    ind = np.array(np.ones(X.shape[1]))
    for i in range(X.shape[1]):
        if "test" not in method:
            if ind[i] > 0:
                for j in range(i + 1, X.shape[1]):
                    if ind[j] > 0:
                        x1 = X[:, i]
                        x2 = X[:, j]

                        if method == "Pearson":
                            rho, pval = stats.pearsonr(x1, x2)
                        elif method == "Spearman":
                            rho, pval = stats.spearmanr(x1, x2, axis=None)

                        # 相关性过高则剔除，可修改阈值
                        ind[j] = 0 if abs(rho) > 0.9 else 1
        else:
            x1 = X[y == 0, i]
            x2 = X[y == 1, i]

            if method == "U-test":
                rho, pval = stats.mannwhitneyu(x1, x2, alternative='two-sided')

            elif method == "T-test":
                equal_var = True if levene(x2, x1)[1] > 0.05 else False
                rho, pval = ttest_ind(x2, x1, equal_var=equal_var)

            ind[i] = 1 if pval < 0.05 else 0

    rand = np.where(ind > 0)[0]
    X_new = X[:, rand]

    return X_new, rand


'''
与标签有关的特征筛选，输出按照重要程度排序的特征索引
KBest:第一个参数为评分函数，可选有f_regression（回归）, mutual_info_regression（连续目标的互信息） 
    mutual_info_classif（离散目标的互信息）,f_classif（分类ANOVA）,chi2（分类任务的非负特征的卡方统计数据），
    分别对应相关性计算，方差分析(ANOVA)，互信息计算，卡方检验
fisher's score:根据Fisher得分以降序返回变量的排名
mutualInformation:method 可以选择 JMI 联合互信息 、JMIM 联合互信息 max、MRMR 最大相关性最小冗余 特征提取方式
disr
relieff:特征权重算法 只适用于二分类问题
MIM:互信息最大化
Lasso:最小绝对收缩和选择算子(Least absolute shrinkage and selection operator)
LR:l2可以保留所有特征 l1可能只保留一个
'''
def feature_selection_with_Label(X, y, method,n_feature,draw=None):
    y=np.asarray(y)
    if n_feature > X.shape[1]:
        n_feature_new = max(min(X.shape[1] // 2,300),min(X.shape[1]-1,51))
        print(f"设定的特征参数{n_feature}大于等于输入的特征数{X.shape[1]}，改为{n_feature_new}")
        n_feature=n_feature_new

    if method=="Fisher_score":
        rank=fisher_score.fisher_score(X, y,"index")

    elif method=="Disr":
        rank = DISR.disr(X, y)

    elif method == "MIM":
        rank = MIM.mim(X, y)

    elif method == "Relief":
        selectors = ReliefF(n_features_to_select=n_feature, n_neighbors=100)
        selectors.fit_transform(X, y)
        weight=selectors.feature_importances_
        rank = selectors.top_features_

    elif method in ["JMI","JMIM","MRMR"]:
        feat_selector = mifs.MutualInformationFeatureSelector(method=method, k=3, n_features="auto")
        feat_selector.fit(X, y)
        rank = feat_selector.ranking_

    elif method in ["chi2","MIC","ANOVA"]:
        if method=="chi2":
            score_func = chi2
        elif method=="MIC":
            score_func = mutual_info_classif
        else:
            score_func = f_classif

        if len(X[X<0]) and method=="chi2":
            X=preprocessing.MinMaxScaler().fit_transform(X)

        selector = SelectKBest(score_func=score_func,k="all")
        selector.fit_transform(X, y)
        weight=selector.scores_
        rank=np.argsort(-weight)

    elif method in ["Lasso","LR"]:
        if method == "Lasso":
            alpha_range = np.logspace(-10, -2, 200, base=10)
            clf = LassoCV(alphas=alpha_range, tol=1e-5, max_iter=5000, cv=5, selection='random').fit(X, y)
            # 筛选出指定阈值相关性的特征
            weight = np.abs(clf.coef_)
        elif method == "LR":
            clf = LogisticRegression(C=0.001).fit(X, y)
            weight = np.abs(clf.coef_[0])

        # threshold = np.median(importance[np.nonzero(importance)])
        # max_features = len(importance[importance > threshold])
        # importance_idx = np.argsort(-importance)
        # rank = importance_idx[0:max_features]
        rank=np.argsort(-weight) #降序排列

    else:
        return "还未实现所用方法"

    ind = rank[0:n_feature]
    X_New = X[:, ind]
    feature_index = np.asarray(ind).astype(int)

    if draw :
        if method == 'Fisher_score':
            print('Fisher_score 现在用的包无法获得特征score,score返回None')
            return X_New, feature_index, None
        else:
            weight=(weight-weight.min())/(weight.max()-weight.min())
            return X_New,feature_index,weight[ind]
    else:
        return X_New, feature_index,None

