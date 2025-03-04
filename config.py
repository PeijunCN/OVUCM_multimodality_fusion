import os
import numpy as np
from pathlib import Path
import warnings
import shutil
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

Program_Path=Path(__file__).resolve().parent.parent #./Ovarian_Cancer_Project

DATA_DIR = os.path.join(Program_Path,"data/data_use")
clinical_and_label_path = os.path.join(DATA_DIR,"clinical_lables_all_NESTCV.xlsx")

cli_withca = [ ] # list the features of clinic data
sheet_name = { } # list the task names

#Feature selection methods
Method = {
    "Feature": ["Spearman", "Pearson", "U-test", "T-test"],
    "Label": ["ANOVA", "Fisher_score", "chi2", "MIC", "Relief", "MRMR", "JMI", "Disr", "Lasso", "LR"],
    "Reduction": ["PCA", "LDA", "LLE"]
}

# parameters of classifiers to chosse in cross-validation
cs = np.logspace(-5, 6, 20, base=2)
vs=np.logspace(-5, 6, 10, base=2)
gammas = np.logspace(-5, 5, 10, base=2)
params_search = {
    "SVC":{"C":cs, "gamma":gammas,"kernel":['rbf', 'sigmoid','linear'],"class_weight":['balanced'], "random_state":[0],"probability":[True],'max_iter': [10000]},
    "LogR":{"penalty": ['l1','l2','elasticnet'],"C":cs,"solver":['liblinear', 'sag', 'saga'],"class_weight":['balanced'], "random_state":[0]},
    "NaiveBayes":{"var_smoothing":vs},
    "XGB":{"learning_rate" : np.arange(0.01,0.7,0.05),"gamma":gammas},
    "LinR":{"fit_intercept":[True,False],"n_jobs":[1,3,5]},
}
