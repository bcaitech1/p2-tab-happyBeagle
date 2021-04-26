# Suppress warnings 
import warnings
import sys
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse
from importlib import import_module

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from pycaret.classification import *

# Custom library
from utils import seed_everything, print_score
from dataset import *
from dataprocess import *

MDOEL_PARAMS = {
    "lgbm": {
            'learning_rate': 0.1,
            'boosting_type': 'gbdt', #‘gbdt’, traditional Gradient Boosting Decision Tree. 
                                     #‘dart’, Dropouts meet Multiple Additive Regression Trees.
                                     #‘goss’, Gradient-based One-Side Sampling. 
                                     # ‘rf’, Random Forest.
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_lambda': 0.2,
            },
    "catboost": {
            'iterations': 5000,
            'loss_function': 'Logloss',
            'task_type': 'GPU',
            'eval_metric': 'AUC',
            'random_seed': SEED,
            'od_type': 'Iter',
            'early_stopping_rounds': 300,
            'learning_rate': 0.07,
            'depth': 8,
            'random_strength': 0.5,
            'verbose': 100,
            'metric_period': 50
            },
    "xgboost": {
            'n_estimators': 200,
            'n_job': -1,
            'max_depth': 8,
            'learning_rate': 0.05,
            'colsample_bytree': 0.5,
            'tree_method': 'gpu_hist'
           }
    }

SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

   
def split_eval(train, labels, x_val, y_val, test, clf, params, fit_params, name):
    scores = []
    feature_importances = np.zeros(len(train.columns))
    test_predictions = np.zeros(test.shape[0])
    test_probablity = np.zeros(test.shape[0])
    
    clf.fit(train, labels, eval_set=[(x_val, y_val)], **fit_params)
    if 'catboost' in name:
        scores.append(clf.best_score_['validation']['AUC'])
    if 'xgboost' in name:
        try:
            scores.append(clf.best_score)
        except:
            scores.append({'valid_0': {'auc': clf.evals_result()['validation_0']['auc'][-1]}})
    if 'lightgbm' in name:
        scores.append(clf.best_score_)

    test_predicts = clf.predict_proba(test)
    test_predictions = test_predicts[:, 1]
    test_probablity = test_predicts[:, 0]
    feature_importances = clf.feature_importances_
    print('-'*60)
    if 'lightgbm' in name:
        scores = [dict(s)['valid_0']['auc'] for s in scores]
    del clf
    return test_predictions, test_probablity, scores, feature_importances

def plot_feature_importances(fe, cols):
    fe = pd.DataFrame(fe, index=cols)
    if fe.shape[1] > 1:
        fe = fe.apply(sum, axis=1)
    else:
        fe = fe[0]
    fe.sort_values(ascending=False)[:20].plot(kind='bar')

def eval_catboost(train, labels, test, params, cat_features, name, eval_set=None):
    clf = CatBoostClassifier(**params)
    fit_params = {
        'cat_features': cat_features,
        'plot':False
    }
    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'catboost_' + name)

def eval_xgboost(train, labels, test, params, name, eval_set=None):
    clf = XGBClassifier(**params)
    fit_params = {
        'verbose':100, 
        'eval_metric':'auc',
        'early_stopping_rounds': 300
    }
    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'xgboost_' + name)

def eval_lightgbm(train, labels, test, params, cat_features, name, eval_set=None):
    clf = LGBMClassifier(**params)
    fit_params = {
        'verbose': 100,
        'eval_metric': 'auc',
        #'categorical_feature':cat_features,        
        'early_stopping_rounds': 300
    }
    
    return split_eval(train, labels, eval_set[0], eval_set[1], test, clf, params, fit_params, 'lightgbm_' + name)


if __name__ == "__main__":
    '''
    reference :  https://www.kaggle.com/gautham11/catboost-xgboost-lightgbm-ensemble/notebook
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_method", type=str, default="ML", help="You can choose ML or DL (default: ML")
    '''
    If you choose ML, you can train data by tabular data
    And then if you choose DL, you can train data which converted to image
    '''
    parser.add_argument("--model", type=str, default="lgbm", help="select model")
    parser.add_argument("--output_path", type=str, help="set output dir path")
    parser.add_argument("--trainset_path", type=str, help="set train data file path .csv")
    parser.add_argument("--testset_path", type=str, help="set test data file path .csv")
    parser.add_argument("--outputform_path", type=str, help="set ouput form data path .csv")

    args = parser.parse_args()
    debug_flag = True

    if debug_flag:
        model = "ensamble"
        trainset_path = "/opt/ml/code/my_src/data/train_data_thres_rate_3_6_12.csv"
        testset_path = "/opt/ml/code/my_src/data/test_data_thres_rate_3_6_12.csv"
        outputform_path = "/opt/ml/code/input/sample_submission.csv"
        output_path = "/opt/ml/code/output/my_run_try_ver6.csv"  
    else:
        model = args.model
        testset_path = args.testset_path 
        trainset_path = args.trainset_path
        outputform_path = args.outputform_path
        output_path = args.output_path
    
    print("-" * 30 + "[dir_info]" + "-" * 30)
    print("ouput path: \t" + output_path)
    print("trainset_path: \t" + trainset_path)
    print("testset_path: \t" + testset_path)
    print("ouputform_path: \t" + outputform_path)
    print("-"*71)
    print("-" * 29 + "[train_info]" + "-" * 29)
    print("model: \t\t" + model)
    print("-" * 71)

    train_data = pd.read_csv(trainset_path)
    test_data = pd.read_csv(testset_path)

    test_data = normalizeData(test_data)
    train_data = normalizeData(train_data)

    features = [feature for feature in test_data.columns if feature != "customer_id"]
    label = train_data["label"]

    train_data = train_data.drop(["customer_id", "label"], axis=1)
    test_data = test_data.drop("customer_id", axis=1)
    
    sub_form = pd.read_csv(outputform_path)

    train_feature, val_feature, train_label, val_label = train_test_split(train_data, label, shuffle=True)
    
    if model == "lgbm":
        model_params = MDOEL_PARAMS[model]
        test_pred, test_probility_cal, score, feature_importances = eval_lightgbm(train_feature, 
                                                                                   train_label, 
                                                                                   test_data, 
                                                                                   model_params, 
                                                                                   ["first_bought", "last_bought"], 
                                                                                   name ="lightgbm_tts", 
                                                                                   eval_set=(val_feature, val_label))
        score_list = test_pred
    elif model == "catboost":
        model_params = MDOEL_PARAMS[model]
        test_pred, test_probility_cal, score, feature_importances = eval_catboost(train_feature, 
                                                                                   train_label, 
                                                                                   test_data, 
                                                                                   model_params, 
                                                                                   ["first_bought", "last_bought"], 
                                                                                   name ="catboost_tts", 
                                                                                   eval_set=(val_feature, val_label))
        score_list = test_pred
    elif model == "xgboost":
        model_params = MDOEL_PARAMS[model]
        test_pred, test_probility_cal, score, feature_importances = eval_xgboost(train_feature, 
                                                                                   train_label, 
                                                                                   test_data, 
                                                                                   model_params, 
                                                                                   name ="xgboost_tts", 
                                                                                   eval_set=(val_feature, val_label))
        score_list = test_pred
    elif model == "ensamble":
        test_pred_lgbm, test_probility_cal_lgbm, score_lgbm, feature_importances_lgbm = eval_lightgbm(train_feature, 
                                                                                                      train_label, 
                                                                                                      test_data, 
                                                                                                      MDOEL_PARAMS["lgbm"], 
                                                                                                      ["first_bought", "last_bought"], 
                                                                                                      name ="lightgbm_tts", 
                                                                                                      eval_set=(val_feature, val_label))
        test_pred_catboost, test_probility_cal_catboost, score_catboost, feature_importances_catboost = eval_catboost(train_feature, 
                                                                                                                      train_label, 
                                                                                                                      test_data, 
                                                                                                                      MDOEL_PARAMS["catboost"], 
                                                                                                                      ["first_bought", "last_bought"], 
                                                                                                                      name ="catboost_tts", 
                                                                                                                      eval_set=(val_feature, val_label))
        test_pred_xgboost, test_probility_cal_xgboost, score_xgboost, feature_importances_xgboost = eval_xgboost(train_feature, 
                                                                                                                 train_label, 
                                                                                                                 test_data, 
                                                                                                                 MDOEL_PARAMS["xgboost"], 
                                                                                                                 name ="xgboost_tts", 
                                                                                                                 eval_set=(val_feature, val_label))

        score_list = (test_pred_catboost + test_pred_lgbm + test_pred_xgboost) / 3
        feature_importances = (feature_importances_catboost + feature_importances_lgbm + feature_importances_xgboost) / 3
        
    else:
        print(f"You choose model that doesn't allow this code. {args.model}")
        print("So we wiil do automl...")
        clf = setup(data=train, target='label', feature_selection=True)
        best3 = compare_models(sort='AUC', n_select=3)
        blended = blend_models(estimator_list = best3, fold = 10, method = 'soft')
        pred_holdout = predict_model(blended)
        final_model = finalize_model(blended)
        predictions = predict_model(final_model, data = test)
        score_list = []
        
        for i, data in sub.iterrows():
            custormer_id = data.customer_id
            temp = predictions[predictions.customer_id == custormer_id]
            score = float(temp["Score"])
            if int(temp["Label"]) == 0:
                score = 1 - score
            score_list.append(score)

    if feature_importances is not None:
        feature_list = []
        for i, f in enumerate(feature_importances):
            feature_list.append((f, i))
        feature_list = sorted(feature_list, key=lambda x : x[0], reverse=True)
        for idx, (f, i) in enumerate(feature_list):
            print(f"{idx} : --> {train_feature.columns[i]}:{f}")

    sub_form["probability"] = score_list
    sub_form.to_csv(output_path, index=False)