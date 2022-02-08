import numpy
import pandas
import sklearn.model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import frappe.FrappeRanker as frappe
from utils.frappe_utils import current_ms, is_ascending


class FrappeInstance:

    def __init__(self):
        self.aggregators = []
        self.calculators = []
        self.dataframe = {}

    def add_calculator(self, calculator):
        self.calculators.append(calculator)

    def add_all_calculators(self):
        self.add_calculator(frappe.RSquaredRanker())
        self.add_calculator(frappe.CosineSimilarityRanker())
        self.add_calculator(frappe.SpearmanRanker())
        self.add_calculator(frappe.ChiSquaredRanker())
        self.add_calculator(frappe.PearsonRanker())
        self.add_calculator(frappe.MutualInfoRanker())
        self.add_calculator(frappe.ANOVARanker())
        self.add_calculator(frappe.SURFRanker())
        self.add_calculator(frappe.ReliefFRanker(n_neighbours=10))
        self.add_calculator(frappe.SURFStarRanker())
        self.add_calculator(frappe.MultiSURFRanker())

    def compute_ranks(self, dataset_name, dataset, label, verbose=True, store=True):
        # Compute Ranks
        ranks = {}
        for calculator in self.calculators:
            start_ms = current_ms()
            ranks[calculator.get_ranker_name()] = \
                pandas.Series(data=calculator.compute_rank(dataset, label), index=dataset.columns)
            if verbose:
                print(calculator.get_ranker_name() + " calculated in " + str(current_ms() - start_ms) + " ms")

        # Aggregate Ranks
        agg_ranks = {}
        start_ms = current_ms()
        for calc_key in ranks.keys():
            agg_dict = {}
            is_asc = is_ascending(calc_key)
            rank_list = ranks[calc_key]
            for aggregator in self.aggregators:
                agg_dict[aggregator.get_name()] = aggregator.calculate_aggregation(rank_list, ascending=is_asc)
            agg_ranks[calc_key] = agg_dict
        if verbose:
            print(str(len(self.aggregators)) + " aggregators calculated in " + str(current_ms() - start_ms) + " ms")
        if store:
            self.update_dataframe(agg_ranks, dataset_name)
        return ranks, agg_ranks

    def compute_classification_score(self, dataset_name, x, y,
                                     classifier=RandomForestClassifier(n_estimators=10),
                                     verbose=True, store=True):
        metric_scores = {}
        x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(x, y, test_size=0.5)
        start_ms = current_ms()
        classifier.fit(x_tr, y_tr)
        y_pred = classifier.predict(x_te)
        metric_scores['accuracy'] = metrics.accuracy_score(y_te, y_pred)
        metric_scores['precision'] = metrics.precision_score(y_te, y_pred)
        metric_scores['recall'] = metrics.recall_score(y_te, y_pred)
        metric_scores['f1'] = metrics.accuracy_score(y_te, y_pred)
        metric_scores['f2'] = metrics.fbeta_score(y_te, y_pred, beta=2)
        metric_scores['auc'] = metrics.roc_auc_score(y_te, y_pred)
        metric_scores['mcc'] = metrics.matthews_corrcoef(y_te, y_pred)
        if verbose:
            print("Classifier train/val in " + str(current_ms() - start_ms) + " ms with accuracy " +
                  str(metric_scores['accuracy']))
        if store:
            self.update_dataframe_scores(metric_scores, dataset_name)
        return metric_scores

    def add_aggregator(self, aggregator):
        self.aggregators.append(aggregator)

    def update_dataframe(self, agg_ranks, dataset_name):
        if not isinstance(self.dataframe, pandas.DataFrame):
            tag_list = ["dataset_name"] + [k + "_" + j for k in agg_ranks for j in agg_ranks[k]]
            self.dataframe = pandas.DataFrame(columns=tag_list)
        self.dataframe.loc[len(self.dataframe), "dataset_name"] = dataset_name
        for dict_tag in agg_ranks:
            for item_tag in list(agg_ranks[dict_tag]):
                tag = dict_tag + "_" + item_tag
                self.dataframe.loc[self.dataframe.dataset_name == dataset_name, tag] = agg_ranks[dict_tag][item_tag]


    def update_dataframe_scores(self, metric_scores, dataset_name):
        if not isinstance(self.dataframe, pandas.DataFrame):
            tag_list = ["dataset_name"] + [k for k in metric_scores]
            self.dataframe = pandas.DataFrame(columns=tag_list)
        for metric in metric_scores:
            self.dataframe.loc[self.dataframe.dataset_name == dataset_name, metric] = metric_scores[metric]

    def print_csv(self, file_name):
        if self.dataframe is not None:
            self.dataframe.to_csv(file_name, index=False)
