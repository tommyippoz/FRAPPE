import numpy
import pandas
import sklearn.model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, LogisticRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import frappe.FrappeRanker as frappe
from utils import frappe_utils
from utils.frappe_utils import current_ms, is_ascending


class FrappeInstance:

    def __init__(self) -> object:
        self.aggregators = []
        self.calculators = []
        self.dataframe = {}

    def add_calculator(self, calculator):
        self.calculators.append(calculator)

    def add_statistical_calculators(self):
        self.add_calculator(frappe.RSquaredRanker())
        self.add_calculator(frappe.CosineSimilarityRanker())
        self.add_calculator(frappe.SpearmanRanker())
        self.add_calculator(frappe.ChiSquaredRanker())
        self.add_calculator(frappe.PearsonRanker())
        self.add_calculator(frappe.MutualInfoRanker())
        self.add_calculator(frappe.ANOVARanker())

    def add_relief_calculators(self):
        self.add_calculator(frappe.SURFRanker())
        self.add_calculator(frappe.ReliefFRanker(n_neighbours=10))
        self.add_calculator(frappe.ReliefFRanker(n_neighbours=20))
        self.add_calculator(frappe.ReliefFRanker(n_neighbours=50))
        self.add_calculator(frappe.SURFStarRanker())
        self.add_calculator(frappe.MultiSURFRanker())

    def add_all_calculators(self):
        self.add_statistical_calculators()
        self.add_relief_calculators()

    def compute_ranks(self, dataset_name, dataset, label, verbose=True, store=True):
        # Compute Ranks
        ranks = {}
        for calculator in self.calculators:
            start_ms = current_ms()
            try:
                calc_rank = calculator.compute_rank(dataset, label)
            except Exception as e:
                print(e)
                calc_rank = numpy.zeros(len(dataset.columns))
            ranks[calculator.get_ranker_name()] = pandas.Series(data=calc_rank, index=dataset.columns)
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
                                     classifiers=[RandomForestClassifier(n_estimators=10)],
                                     verbose=True, store=True):

        metric_scores = frappe_utils.classification_analysis(x, y, classifiers, verbose)
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

    def regression_analysis(self, target_metric, train_split=0.66, select_features=None, data_augmentation=False, verbose=True):
        return frappe_utils.regression_analysis(start_df=self.dataframe, label_tag=target_metric, train_split=train_split,
                                                regressors=[RandomForestRegressor(n_estimators=10),
                                                            RandomForestRegressor(n_estimators=100),
                                                            XGBRegressor(n_estimators=10),
                                                            XGBRegressor(n_estimators=100),
                                                            XGBRegressor(n_estimators=500)],
                                                select_features=select_features,
                                                verbose=verbose,
                                                data_augmentation=data_augmentation)

    def load_dataframe(self, df):
        self.dataframe = df
