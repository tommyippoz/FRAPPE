import os

import frappe.FrappeRanker as fr
import pandas
from frappe.FrappeAggregator import GetBest, GetAverage, GetAverageBest
from frappe.FrappeInstance import FrappeInstance
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset
from utils.frappe_utils import current_ms
from xgboost import XGBClassifier

INPUT_FOLDER = "input_folder"
OUTPUT_FOLDER = "output_folder"

LABEL_NAME = "multilabel"
OUTPUT_FILE = "timings_unsup.csv"


def get_supervised_classifiers():
    return [GaussianNB(),
            BernoulliNB(),
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            RandomForestClassifier(),
            LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=10)]


def get_unsupervised_classifiers(outliers_fraction):
    return [GridSearchCV(estimator=KMeans(n_jobs=-1), scoring='roc_auc',
                         param_grid={'n_clusters': [2, 5, 10, 20, 50, 100]}),
            COPOD(contamination=outliers_fraction, n_jobs=-1),
            IsolationForest(warm_start=True, n_jobs=-1),
            ABOD(contamination=outliers_fraction, method='fast', n_jobs=-1),
            #GridSearchCV(estimator=ABOD(contamination=outliers_fraction, method='fast', n_jobs=-1), scoring='roc_auc',
            #             param_grid={'n_neighbors': [1, 3, 5]}),
            GridSearchCV(estimator=HBOS(contamination=outliers_fraction, n_jobs=-1), scoring='roc_auc',
                         param_grid={'n_bins': [5, 10, 20, 50, 100, 200], 'tol': [0.2, 0.5, 0.8]}),
            GridSearchCV(estimator=MCD(contamination=outliers_fraction, n_jobs=-1), scoring='roc_auc',
                         param_grid={'support_fraction': [None, 0.1, 0.3, 0.5]}),
            GridSearchCV(estimator=PCA(contamination=outliers_fraction, n_jobs=-1), scoring='roc_auc',
                         param_grid={'weighted': [False, True]})]

 
if __name__ == '__main__':

    pandas.set_option('mode.chained_assignment', None)

    frappe = FrappeInstance()
    frappe.add_calculator(fr.MutualInfoRanker())

    # frappe.add_statistical_calculators()
    # frappe.add_calculator(fr.ReliefFRanker(n_neighbours=10, limit_rows=2000))
    # frappe.add_calculator(fr.SURFRanker(limit_rows=2000))
    # frappe.add_calculator(fr.MultiSURFRanker(limit_rows=2000))
    #frappe.add_calculator(fr.CoefRanker(LinearRegression()))
    # frappe.add_calculator(fr.WrapperRanker(RandomForestClassifier(n_estimators=10)))
    # frappe.add_calculator(fr.WrapperRanker(FastAI(label_name=LABEL_NAME)))
    #
    frappe.add_aggregator(GetBest())
    frappe.add_aggregator(GetAverageBest(n=3))
    frappe.add_aggregator(GetAverageBest(n=5))
    frappe.add_aggregator(GetAverageBest(n=10))
    frappe.add_aggregator(GetAverage())

    dataset_files = frappe_utils.get_dataset_files(INPUT_FOLDER)

    timings_df = 0

    for file_name in tqdm(dataset_files, desc="Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset(file_name, LABEL_NAME)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nProcessing tag " + tag + " of dataset " + dataset_name)

            ranks, agg_ranks, timings = frappe.compute_ranks(dataset_name + "@" + tag, x, y, store=True)
            start_ms = current_ms()
            classifiers = get_supervised_classifiers()
            compute_classification_score(dataset_name + "@" + tag, x, y, store=True,
                                                classifiers=[DecisionTreeClassifier()])
            timings["Avg Classifier Time"] = (current_ms() - start_ms)/len(classifiers)
            timings["# Classifiers"] = len(classifiers)
            timings["dataset_name"] = dataset_name
            timings["dataset_size"] = len(y)
            timings["n_features"] = x.shape[1]
            timings["train_size"] = int(len(y)/2)

            if not isinstance(timings_df, pandas.DataFrame):
                timings_df = pandas.DataFrame(columns=list(timings.keys()))
            timings_df = timings_df.append(timings, ignore_index=True)

    timings_df.to_csv(OUTPUT_FOLDER + "/" + OUTPUT_FILE, index=False)
