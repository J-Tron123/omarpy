"""El modulo :mod:omarpy.machine_learning incluye funciones que ayudan con el entrenamiento de modelos predictivos de machine learning. """


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from sklearn.model_selection import train_test_split

from prettytable import PrettyTable

import pickle
import sys
import os
import pandas as pd
import numpy as np
import urllib.request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
import xgboost
import re
from nltk.text import stopwords

from .machine_learning import (
Mejor_PCA_DecissionTree_Regression,
Mejor_PCA_RandomForest_Regression,
Mejor_PCA_XGB_Regression,
scaler,
prediction,
class_results,
binary_class_metrics,
precision_recall_AUC,
load_model,
XgBoost_X_y,
optunaXGBOOST,
omar,
scores,
similarity_index,
step_axis,
percentil,
metodo_iqr,
metodo_std,
DF_Feature_importance,
preprocess_reviews,
remove_stop_words,
sweet_table,
check_optimizadores,)

