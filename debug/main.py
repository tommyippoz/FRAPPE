import os

import frappe.FrappeRanker as fr
from frappe.FrappeAggregator import GetBest, GetAverage, GetAverageBest, GetSum
from frappe.FrappeInstance import FrappeInstance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset
from xgboost import XGBClassifier

INPUT_FOLDER = "G:/My Drive/Datasets_CriticalSystems/scikit"
OUTPUT_FOLDER = "output_folder"
LABEL_NAME = "multilabel"

if __name__ == '__main__':

    frappe = FrappeInstance()

    # frappe.add_all_calculators()
    frappe.add_statistical_calculators()
    frappe.add_calculator(fr.ReliefFRanker(n_neighbours=10))
    frappe.add_calculator(fr.ReliefFRanker(n_neighbours=20))
    frappe.add_calculator(fr.WrapperRanker(RandomForestClassifier(n_estimators=10)))
    #frappe.add_calculator(fr.WrapperRanker(XGBClassifier(use_label_encoder=False)))
    #frappe.add_calculator(fr.WrapperRanker(TabNetClassifier()))
    #frappe.add_calculator(fr.WrapperRanker(FastAI(LABEL_NAME)))

    frappe.add_aggregator(GetBest())
    frappe.add_aggregator(GetAverageBest(n=3))
    frappe.add_aggregator(GetAverageBest(n=5))
    frappe.add_aggregator(GetAverageBest(n=10))
    frappe.add_aggregator(GetAverage())
    frappe.add_aggregator(GetSum())

    dataset_files = frappe_utils.get_dataset_files(INPUT_FOLDER)

    for file_name in dataset_files:

        print("\n------------------------------------------------------\n")
        x, y, feature_names, label_names = load_binary_tabular_dataset(file_name, LABEL_NAME)

        dataset_name = os.path.basename(file_name).replace(".csv", "")
        ranks, agg_ranks = frappe.compute_ranks(dataset_name, x, y, store=True)
        compute_classification_score(dataset_name, x, y, store=True,
                                            classifiers=[GaussianNB(),
                                                         BernoulliNB(),
                                                         RandomForestClassifier(n_estimators=10),
                                                         XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                                                         LinearDiscriminantAnalysis(),
                                                         ExtraTreesClassifier(n_estimators=10)])



    frappe.print_csv("output_folder/frappe_frame.csv")



