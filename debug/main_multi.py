import os

import pandas
from logitboost import LogitBoost
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from src.FrappeInstance import FrappeInstance
from src.FrappeRanker import WrapperRanker, CoefRanker
from src.FrappeType import FrappeType
from src.dataset_utils import load_binary_tabular_dataset_array_partition
from src.frappe_utils import get_dataset_files, compute_classification_score

INPUT_FOLDER = "../input"
OUTPUT_FOLDER = "../output"

LABEL_NAME = "multilabel"
OUTPUT_FILE = "all_calc_uns.csv"


def get_supervised_classifiers():
    return [Pipeline([('scaler', MinMaxScaler()), ('gnb', GaussianNB())]),
            DecisionTreeClassifier(),
            LogisticRegression(),
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            LinearDiscriminantAnalysis(),
            ExtraTreesClassifier(n_estimators=10),
            LogitBoost(n_estimators=30)]


def get_unsupervised_classifiers(outliers_fraction):
    return [KMeans(n_clusters=10, n_init='auto'),
            COPOD(contamination=outliers_fraction),
            IsolationForest(n_estimators=10, contamination=outliers_fraction),
            HBOS(contamination=outliers_fraction),
            MCD(contamination=outliers_fraction, support_fraction=0.9),
            PCA(contamination=outliers_fraction)]


if __name__ == '__main__':

    pandas.set_option('mode.chained_assignment', None)

    existing_exps = None
    if os.path.exists(OUTPUT_FILE):
        existing_exps = pandas.read_csv(OUTPUT_FILE)
        existing_exps = existing_exps.loc[:, ['dataset_name']]

    frappe = FrappeInstance(load_models=False, models_folder="../models",
                            instance=FrappeType.CUSTOM,
                            custom_rankers=[CoefRanker(LinearRegression())])
    dataset_files = get_dataset_files(INPUT_FOLDER)

    for file_name in tqdm(dataset_files, desc="Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset_array_partition(file_name, LABEL_NAME, n_partitions=5)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            full_name = dataset_name + "@" + tag

            if existing_exps is not None and ((existing_exps['dataset_name'] == full_name).any()):
                print('Skipping dataset partition %s, already in the results' % (full_name))
            else:
                print("\nProcessing tag " + tag + " of dataset " + dataset_name)
                ranks = frappe.compute_ranks(dataset_x=x, dataset_y=y, dataset_name=full_name)
                metrics_df, metric_scores = \
                    compute_classification_score(full_name, x, y,
                                                 classifiers=get_unsupervised_classifiers(an_perc if an_perc < 0.5 else 0.5))

                full_df = pandas.merge(ranks, metrics_df, on="dataset_name")
                if not os.path.exists(OUTPUT_FILE):
                    full_df.to_csv(OUTPUT_FILE, index=False)
                else:
                    full_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

    frappe.print_csv(OUTPUT_FOLDER + "/" + OUTPUT_FILE)

    [metric_scores, x_te, y_te, y_pred] = frappe.regression_analysis(target_metric="mcc")
    x_te["true_value"] = y_te
    x_te["predicted_value"] = y_pred
    x_te.to_csv(OUTPUT_FOLDER + "/" + OUTPUT_FILE.replace(".csv", "_RegTest.csv"), index=False)
