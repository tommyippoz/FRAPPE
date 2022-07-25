import os
import time
from os.path import isdir

import numpy
import pandas
import sklearn
import smogn
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import cross_val_score

USED_METRICS = ["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "f2", "auc", "mcc"]


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def is_ascending(calc_key):
    if calc_key in {"CosineSimilarity"}:
        return True
    else:
        return False


def get_dataset_files(folder, partial_list=[]):
    for dataset_file in os.listdir(folder):
        full_name = folder + "/" + dataset_file
        if full_name.endswith(".csv"):
            partial_list.append(full_name)
        elif isdir(full_name):
            partial_list = get_dataset_files(full_name, partial_list)
    return partial_list


def classification_analysis(x, y, classifiers, verbose=True):

    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(x, y, test_size=0.5)

    best_result = [-1, []]
    for classifier in classifiers:
        start_ms = current_ms()
        classifier.fit(x_tr, y_tr)
        y_pred = classifier.predict(x_te)
        mcc = metrics.matthews_corrcoef(y_te, y_pred)
        if mcc < 0:
            mcc = - mcc
            y_pred = 1 - y_pred
        if mcc > best_result[0]:
            best_result = [mcc, y_pred]
        if verbose:
            print("Classifier " + classifier.__class__.__name__ + " train/val in " + str(current_ms() - start_ms) +
                  " ms with MCC of " + str(mcc))
    a = numpy.unique(best_result[1])
    has_single = len(numpy.unique(best_result[1])) == 1
    tn, fp, fn, tp = metrics.confusion_matrix(y_te, best_result[1]).ravel()
    metric_scores = {'tp': tp,
                     'tn': tn,
                     'fp': fp,
                     'fn': fn,
                     'accuracy': metrics.accuracy_score(y_te, best_result[1]),
                     'precision': metrics.precision_score(y_te, best_result[1]),
                     'recall': metrics.recall_score(y_te, best_result[1]),
                     'f1': metrics.accuracy_score(y_te, best_result[1]) if not has_single else 0.0,
                     'f2': metrics.fbeta_score(y_te, best_result[1], beta=2) if not has_single else 0.0,
                     'auc': metrics.roc_auc_score(y_te, best_result[1]) if not has_single else 0.0,
                     'mcc': metrics.matthews_corrcoef(y_te, best_result[1]) if not has_single else 0.0}
    if verbose:
        print("Best classifier gets MCC of " + str(metric_scores['mcc'])
              + " and accuracy of " + str(metric_scores['accuracy']))

    return metric_scores


def get_feature_importances(algorithm):
    if hasattr(algorithm, 'feature_importances_'):
        return algorithm.feature_importances_
    elif  hasattr(algorithm, 'coef_'):
        return algorithm.coef_
    else:
        return 0


def regression_analysis(start_df, label_tag, train_split, regressors, select_features=None, verbose=True, data_augmentation=False):

    df = clear_regression_dataframe(reg_df=start_df, target_metric=label_tag, select_features=select_features)

    x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(
        df.drop(columns=[label_tag]), df[label_tag], test_size=1-train_split)

    if data_augmentation:
        train_df = x_tr.copy()
        train_df[label_tag] = y_tr
        train_df.reset_index(inplace=True, drop=True)
        print("Data Augmentation with SMOTER")
        train_smogn = smogn.smoter(data=train_df, y=label_tag, rel_coef=0.2)
        if verbose:
            print("Data Augmentation performed: generated " + str(len(train_smogn.index)) + " additional train rows.")
        train_smogn = pandas.concat([train_smogn, train_df])
        y_tr = train_smogn[label_tag]
        x_tr = train_smogn.drop(columns=[label_tag])

    best_result = [100000, []]
    for regressor in regressors:
        start_ms = current_ms()
        regressor.fit(x_tr, y_tr)
        y_pred = regressor.predict(x_te)
        mae = metrics.mean_absolute_error(y_te, y_pred)
        if mae < best_result[0]:
            imp = get_feature_importances(regressor)
            best_result = [mae, y_pred, dict(zip(x_te.columns, imp)), regressor]
        if verbose:
            print("Regressor " + regressor.__class__.__name__ + " train/val in " + str(current_ms() - start_ms) +
                  " ms with mean absolute error of " + str(mae) + " and R2 of " + str(metrics.r2_score(y_te, y_pred)))
    metric_scores = {'exp_var': metrics.explained_variance_score(y_te, best_result[1]),
                     'max_err': metrics.max_error(y_te, best_result[1]),
                     'mean_abs_err': metrics.mean_absolute_error(y_te, best_result[1]),
                     'mean_sq_err': metrics.mean_squared_error(y_te, best_result[1]),
                     'r2': metrics.r2_score(y_te, best_result[1])}
    if verbose:
        print("Best regressor gets Mean Absolute Error of " + str(metric_scores['mean_abs_err'])
              + " and R2 of " + str(metric_scores['r2']))

    x_te = pandas.concat([x_te, start_df], axis=1, join="inner")
    x_te = x_te.loc[:, ~x_te.columns.duplicated()]

    return best_result[3], [metric_scores, dict(sorted(best_result[2].items(), key=lambda item: item[1], reverse=True)), x_te, y_te, best_result[1]]


def clear_regression_dataframe(reg_df, target_metric="mcc", select_features=None):
    y = reg_df[target_metric]
    my_df = reg_df.drop(columns=["dataset_name"])
    my_df = my_df.drop(columns=USED_METRICS)
    if select_features is not None:
        if isinstance(select_features, int):
            kbest = SelectKBest(mutual_info_regression, k=int(select_features))
        else:
            kbest = SelectKBest(mutual_info_regression, k=10)
        kbest.fit_transform(my_df, y)
        my_df = my_df.iloc[:, kbest.get_support()]
    my_df[target_metric] = y
    for column in my_df.columns:
        my_df[column] = my_df[column].astype(float)
    return my_df

