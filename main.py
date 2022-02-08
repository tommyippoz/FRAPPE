import os

import numpy
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from frappe.FrappeAggregator import GetBest, GetAverage, GetAverageBest, GetSum
from frappe.FrappeInstance import FrappeInstance
from frappe.FrappeRanker import SURFRanker, SURFStarRanker, MultiSURFRanker, ReliefFRanker, RSquaredRanker, \
    CosineSimilarityRanker, WrapperRanker
from utils import frappe_utils
from utils.AutoGluonClassifier import FastAI, GBM
from utils.dataset_utils import load_tabular_dataset, load_binary_tabular_dataset

INPUT_FOLDER = "input_folder"
OUTPUT_FOLDER = "output_folder"
LABEL_NAME = "multilabel"

if __name__ == '__main__':

    frappe = FrappeInstance()

    frappe.add_calculator(RSquaredRanker())
    frappe.add_calculator(CosineSimilarityRanker())
    #frappe.add_calculator(WrapperRanker(RandomForestClassifier(n_estimators=10)))
    #frappe.add_calculator(WrapperRanker(XGBClassifier(use_label_encoder=False)))
    #frappe.add_calculator(WrapperRanker(TabNetClassifier()))
    frappe.add_calculator(WrapperRanker(FastAI(LABEL_NAME)))

    frappe.add_aggregator(GetBest())
    frappe.add_aggregator(GetAverageBest(n=3))
    frappe.add_aggregator(GetAverageBest(n=5))
    frappe.add_aggregator(GetAverageBest(n=10))
    frappe.add_aggregator(GetAverage())
    frappe.add_aggregator(GetSum())

    dataset_files = frappe_utils.get_dataset_files(INPUT_FOLDER)

    for file_name in dataset_files:
        x, y, feature_names, label_names = load_binary_tabular_dataset(file_name, LABEL_NAME)

        dataset_name = os.path.basename(file_name).replace(".csv", "")
        ranks, agg_ranks = frappe.compute_ranks(dataset_name, x, y, store=True)
        frappe.compute_classification_score(dataset_name, x, y, store=True)

    frappe.print_csv("output_folder/frappe_frame.csv")



