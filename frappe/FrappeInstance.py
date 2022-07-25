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
from frappe.FrappeAggregator import GetAverageBest, GetAverage, GetSum, GetBest
from utils import frappe_utils
from utils.AutoGluonClassifier import FastAI
from utils.frappe_utils import current_ms, is_ascending

from joblib import dump, load


class FrappeInstance:

    def __init__(self, load_models=True, rankers_set="full") -> object:
        self.aggregators = []
        self.calculators = []
        self.dataframe = {}
        self.regressors = {}
        for ad_type in ["SUP", "UNSUP"]:
            self.regressors[ad_type] = {}
        if load_models:
            for ad_type in ["SUP", "UNSUP"]:
                for metric in ["mcc", "auc"]:
                    if rankers_set is "full":
                        self.load_regression_model(metric, ad_type,
                                                   "models/" + ad_type + "_" + metric + "_model.joblib")
                    else:
                        self.load_regression_model(metric, ad_type,
                                                   "models/" + ad_type + "_" + metric + "_red_model.joblib")

        if rankers_set is "full":
            self.add_default_calculators()
        else:
            self.add_reduced_calculators()
        self.add_default_aggregators()

    def add_default_aggregators(self):
        self.add_aggregator(GetBest())
        self.add_aggregator(GetAverageBest(n=3))
        self.add_aggregator(GetAverageBest(n=5))
        self.add_aggregator(GetAverageBest(n=10))
        self.add_aggregator(GetAverage())
        self.add_aggregator(GetSum())

    def add_default_calculators(self):
        self.add_statistical_calculators()
        self.add_calculator(frappe.ReliefFRanker(n_neighbours=10, limit_rows=2000))
        self.add_calculator(frappe.SURFRanker(limit_rows=2000))
        self.add_calculator(frappe.MultiSURFRanker(limit_rows=2000))
        self.add_calculator(frappe.CoefRanker(LinearRegression()))
        self.add_calculator(frappe.WrapperRanker(RandomForestClassifier(n_estimators=10)))
        self.add_calculator(frappe.WrapperRanker(FastAI(label_name="multilabel")))

    def add_reduced_calculators(self):
        self.add_calculator(frappe.RSquaredRanker())
        self.add_calculator(frappe.SpearmanRanker())
        self.add_calculator(frappe.ChiSquaredRanker())
        self.add_calculator(frappe.PearsonRanker())
        self.add_calculator(frappe.MutualInfoRanker())
        self.add_calculator(frappe.ReliefFRanker(n_neighbours=10, limit_rows=2000))
        self.add_calculator(frappe.WrapperRanker(RandomForestClassifier(n_estimators=10)))
        self.add_calculator(frappe.WrapperRanker(FastAI(label_name="multilabel")))

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
        timings = {}
        for calculator in self.calculators:
            start_ms = current_ms()
            try:
                calc_rank = calculator.compute_rank(dataset, label)
            except Exception as e:
                print(e)
                calc_rank = numpy.zeros(len(dataset.columns))
            ranks[calculator.get_ranker_name()] = pandas.Series(data=calc_rank, index=dataset.columns)
            timings[calculator.get_ranker_name()] = (current_ms() - start_ms)
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
        return ranks, agg_ranks, timings

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

    def regression_analysis(self, target_metric, ad_type, train_split=0.66, select_features=None, data_augmentation=False, verbose=True):
        model, return_array = frappe_utils.regression_analysis(start_df=self.dataframe, label_tag=target_metric,
                                                               train_split=train_split,
                                                               regressors=[RandomForestRegressor(n_estimators=10),
                                                                           RandomForestRegressor(n_estimators=100),
                                                                           XGBRegressor(n_estimators=10),
                                                                           XGBRegressor(n_estimators=100),
                                                                           XGBRegressor(n_estimators=500)],
                                                               select_features=select_features,
                                                               verbose=verbose,
                                                               data_augmentation=data_augmentation)
        self.regressors[ad_type][target_metric] = model
        return return_array

    def load_dataframe(self, df):
        self.dataframe = df

    def load_regression_model(self, metric, ad_type, model_path):
        self.regressors[ad_type][metric] = load(model_path)

    def save_regression_model(self, metric, ad_type, model_path):
        dump(self.regressors[ad_type][metric], model_path)

    def predict_metric(self, metric, ad_type, x, y):
        start_ms = current_ms()
        ranks, agg_ranks, timings = self.compute_ranks(dataset_name=None, dataset=x, label=y, store=False)
        middle_ms = current_ms()
        model = self.regressors[ad_type][metric]
        data = numpy.asarray([d[x] for d in agg_ranks.values() for x in d.keys()])
        data[numpy.isnan(data)] = 0
        pred_metric = model.predict(data.reshape(1, -1))[0]

        return pred_metric, middle_ms - start_ms, current_ms() - middle_ms, data
