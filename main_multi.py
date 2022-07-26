import os

import pandas
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.loci import LOCI
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.suod import SUOD
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

from frappe.FrappeAggregator import GetBest, GetAverage, GetAverageBest, GetSum
from frappe.FrappeInstance import FrappeInstance
import frappe.FrappeRanker as fr
from utils import frappe_utils
from utils.AutoGluonClassifier import FastAI
from utils.dataset_utils import load_binary_tabular_dataset, load_binary_tabular_dataset_array, \
    load_binary_tabular_dataset_array_partition

INPUT_FOLDER = "input_folder"
OUTPUT_FOLDER = "output_folder"

LABEL_NAME = "multilabel"
OUTPUT_FILE = "all_calc_unsup.csv"


def get_supervised_classifiers():
    return [GaussianNB(),
            BernoulliNB(),
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=10)]


def get_unsupervised_classifiers(outliers_fraction):
    return [XGBClassifier(use_label_encoder=False),
            GridSearchCV(estimator=KMeans(), scoring='roc_auc',
                         param_grid={'n_clusters': [2, 5, 10, 20, 50, 100]}),
            COPOD(contamination=outliers_fraction),
            IsolationForest(warm_start=True),
            ABOD(contamination=outliers_fraction, method='fast'),
            GridSearchCV(estimator=ABOD(contamination=outliers_fraction, method='fast'), scoring='roc_auc',
                         param_grid={'n_neighbors': [1, 3, 5]}),
            GridSearchCV(estimator=HBOS(contamination=outliers_fraction), scoring='roc_auc',
                         param_grid={'n_bins': [5, 10, 20, 50, 100, 200], 'tol': [0.2, 0.5, 0.8]}),
            GridSearchCV(estimator=MCD(contamination=outliers_fraction), scoring='roc_auc',
                         param_grid={'support_fraction': [None, 0.1, 0.3, 0.5]}),
            GridSearchCV(estimator=PCA(contamination=outliers_fraction), scoring='roc_auc',
                         param_grid={'weighted': [False, True]})]

 
if __name__ == '__main__':

    pandas.set_option('mode.chained_assignment', None)

    frappe = FrappeInstance()

    # frappe.add_statistical_calculators()
    # frappe.add_calculator(fr.ReliefFRanker(n_neighbours=10, limit_rows=2000))
    # frappe.add_calculator(fr.SURFRanker(limit_rows=2000))
    # frappe.add_calculator(fr.MultiSURFRanker(limit_rows=2000))
    # frappe.add_calculator(fr.WrapperRanker(RandomForestClassifier(n_estimators=10)))
    # frappe.add_calculator(fr.WrapperRanker(FastAI(label_name=LABEL_NAME)))
    #
    # frappe.add_aggregator(GetBest())
    # frappe.add_aggregator(GetAverageBest(n=3))
    # frappe.add_aggregator(GetAverageBest(n=5))
    # frappe.add_aggregator(GetAverageBest(n=10))
    # frappe.add_aggregator(GetAverage())
    # frappe.add_aggregator(GetSum())

    dataset_files = frappe_utils.get_dataset_files(INPUT_FOLDER)

    for file_name in tqdm(dataset_files, desc="Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset(file_name, LABEL_NAME)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nProcessing tag " + tag + " of dataset " + dataset_name)

            ranks, agg_ranks = frappe.compute_ranks(dataset_name + "@" + tag, x, y, store=True)
            compute_classification_score(dataset_name + "@" + tag, x, y, store=True,
                                                classifiers=[COPOD(contamination=an_perc)])

    frappe.print_csv(OUTPUT_FOLDER + "/" + OUTPUT_FILE)

    [metric_scores, x_te, y_te, y_pred] = frappe.regression_analysis(target_metric="mcc")
    x_te["true_value"] = y_te
    x_te["predicted_value"] = y_pred
    x_te.to_csv(OUTPUT_FOLDER + "/" + OUTPUT_FILE.replace(".csv", "_RegTest.csv"), index=False)
