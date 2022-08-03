import os

import numpy
import pandas
from pyod.models.copod import COPOD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

import frappe.FrappeRanker as franker
from frappe.FrappeAggregator import GetAverageBest, GetAverage, GetSum, GetBest
from frappe.FrappeType import FrappeType
from utils import frappe_utils
from utils.AutoGluonClassifier import FastAI
from utils.frappe_utils import current_ms, is_ascending, write_dict, exercise_classifier

from joblib import dump, load


class FrappeInstance:

    def __init__(self, load_models=True, instance=FrappeType.REGULAR,
                 models_folder="models/", custom_rankers=None) -> object:
        self.instance_type = instance
        self.aggregators = []
        self.calculators = []
        self.models_folder = models_folder
        self.regressors = {"SUP": {}, "UNS": {}}
        if load_models:
            self.load_models()
        if self.instance_type is FrappeType.FULL:
            self.add_all_calculators()
        elif self.instance_type is FrappeType.REGULAR:
            self.add_reduced_calculators()
        elif self.instance_type is FrappeType.FAST:
            self.add_fast_calculators()
        elif self.instance_type is FrappeType.CUSTOM:
            [self.add_calculator(ranker) for ranker in custom_rankers]
        self.add_default_aggregators()

    def get_model_file(self, ad_type, metric) -> object:
        model_file = self.models_folder
        if self.instance_type is FrappeType.FULL:
            model_file += "/full/"
        elif self.instance_type is FrappeType.REGULAR:
            model_file += "/regular/"
        elif self.instance_type is FrappeType.FAST:
            model_file += "/fast/"
        elif self.instance_type is FrappeType.CUSTOM:
            model_file += "/custom/"
        if not os.path.exists(model_file):
            os.mkdir(model_file)
        model_file += str(ad_type) + "_" + str(metric) + "_model.joblib"
        return model_file

    def load_models(self):
        for ad_type in list(self.regressors.keys()):
            for metric in ["mcc", "auc"]:
                model_file = self.get_model_file(ad_type, metric)
                if os.path.exists(model_file):
                    self.regressors[ad_type][metric] = load(model_file)
                    print("Loaded model for " + str(self.instance_type) + "/" + str(ad_type))
                else:
                    print("Unable to locate model for " + str(self.instance_type) + "/" + str(ad_type))

    def add_default_aggregators(self):
        self.add_aggregator(GetBest())
        self.add_aggregator(GetAverageBest(n=3))
        self.add_aggregator(GetAverageBest(n=5))
        self.add_aggregator(GetAverageBest(n=10))
        self.add_aggregator(GetAverage())
        self.add_aggregator(GetSum())

    def add_default_calculators(self):
        self.add_statistical_calculators()
        self.add_calculator(franker.ReliefFRanker(n_neighbours=10, limit_rows=2000))
        self.add_calculator(franker.SURFRanker(limit_rows=2000))
        self.add_calculator(franker.MultiSURFRanker(limit_rows=2000))
        self.add_calculator(franker.CoefRanker(LinearRegression()))
        self.add_calculator(franker.WrapperRanker(RandomForestClassifier(n_estimators=10)))
        self.add_calculator(franker.WrapperRanker(FastAI(label_name="multilabel", )))

    def add_reduced_calculators(self):
        self.add_calculator(franker.RSquaredRanker())
        self.add_calculator(franker.SpearmanRanker())
        self.add_calculator(franker.ChiSquaredRanker())
        self.add_calculator(franker.PearsonRanker())
        self.add_calculator(franker.MutualInfoRanker())
        self.add_calculator(franker.ReliefFRanker(n_neighbours=10, limit_rows=2000))
        self.add_calculator(franker.WrapperRanker(RandomForestClassifier(n_estimators=10)))
        self.add_calculator(franker.WrapperRanker(FastAI(label_name="multilabel")))

    def add_fast_calculators(self):
        self.add_calculator(franker.RSquaredRanker())
        self.add_calculator(franker.SpearmanRanker())
        self.add_calculator(franker.ChiSquaredRanker())
        self.add_calculator(franker.PearsonRanker())
        self.add_calculator(franker.WrapperRanker(RandomForestClassifier(n_estimators=10)))

    def add_calculator(self, calculator):
        self.calculators.append(calculator)

    def add_statistical_calculators(self):
        self.add_calculator(franker.RSquaredRanker())
        self.add_calculator(franker.CosineSimilarityRanker())
        self.add_calculator(franker.SpearmanRanker())
        self.add_calculator(franker.ChiSquaredRanker())
        self.add_calculator(franker.PearsonRanker())
        self.add_calculator(franker.MutualInfoRanker())
        self.add_calculator(franker.ANOVARanker())

    def add_relief_calculators(self):
        self.add_calculator(franker.SURFRanker())
        self.add_calculator(franker.ReliefFRanker(n_neighbours=10))
        self.add_calculator(franker.ReliefFRanker(n_neighbours=20))
        self.add_calculator(franker.ReliefFRanker(n_neighbours=50))
        self.add_calculator(franker.SURFStarRanker())
        self.add_calculator(franker.MultiSURFRanker())

    def add_all_calculators(self):
        self.add_statistical_calculators()
        self.add_relief_calculators()

    def compute_ranks(self, dataset_x, dataset_y, dataset_name="basic", ranks_df=None, verbose=True):

        if ranks_df is None:
            # Init DataFrame
            tag_list = ["dataset_name"] + [k.get_ranker_name() + "_" + j.get_name()
                                           for k in self.calculators for j in self.aggregators]
            ranks_df = pandas.DataFrame(columns=tag_list)

        # Compute Ranks
        ranks, timings = self.compute_dataset_ranks(dataset_x, dataset_y)
        ranks_df.loc[len(ranks_df), "dataset_name"] = dataset_name
        for dict_tag in ranks:
            for item_tag in list(ranks[dict_tag]):
                tag = dict_tag + "_" + item_tag
                ranks_df.loc[ranks_df.dataset_name == dataset_name, tag] = \
                    ranks[dict_tag][item_tag]

        return ranks_df

    def compute_raw_dataset_ranks(self, dataset, label):
        ranks = {}
        for calculator in self.calculators:
            try:
                calc_rank = calculator.compute_rank(dataset, label)
            except Exception as e:
                print(e)
                calc_rank = numpy.zeros(len(dataset.columns))
            ranks[calculator.get_ranker_name()] = pandas.Series(data=calc_rank, index=dataset.columns)

        return ranks

    def compute_raw_aggregation(self, ranks):
        aggregate = {}
        for calculator in self.calculators:
            calc_rank = ranks[calculator.get_ranker_name()]
            is_asc = is_ascending(calculator.get_ranker_name())
            agg_dict = {}
            for aggregator in self.aggregators:
                agg_dict[aggregator.get_name()] = \
                    aggregator.calculate_aggregation(calc_rank, ascending=is_asc)
            aggregate[calculator.get_ranker_name()] = agg_dict
        agg_row = [aggregate[calculator.get_ranker_name()][aggregator.get_name()]
                   for calculator in self.calculators for aggregator in self.aggregators]

        return aggregate, agg_row

    def compute_dataset_ranks(self, dataset, label, verbose=True):

        ranks = {}
        timings = {}
        for calculator in self.calculators:

            # Compute Ranks
            start_ms = current_ms()
            try:
                calc_rank = calculator.compute_rank(dataset, label)
            except Exception as e:
                print(e)
                calc_rank = numpy.zeros(len(dataset.columns))
            calc_rank = pandas.Series(data=calc_rank, index=dataset.columns)
            timings[calculator.get_ranker_name()] = (current_ms() - start_ms)

            # Aggregate
            is_asc = is_ascending(calculator.get_ranker_name())
            agg_dict = {}
            for aggregator in self.aggregators:
                agg_dict[aggregator.get_name()] = \
                    aggregator.calculate_aggregation(calc_rank, ascending=is_asc)
            ranks[calculator.get_ranker_name()] = agg_dict

            if verbose:
                print(calculator.get_ranker_name() + " calculated in " + str(current_ms() - start_ms) + " ms")

        return ranks, timings

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

    def print_csv(self, file_name):
        if self.dataframe is not None:
            self.dataframe.to_csv(file_name, index=False)

    def learn_models(self, ranks_df, ad_type, train_split=0.66,
                     select_features=None, data_augmentation=False, verbose=True):

        # Learning Models
        for metric in ["mcc", "auc"]:
            model, return_array = \
                frappe_utils.regression_analysis(start_df=ranks_df, label_tag=metric,
                                                 train_split=train_split,
                                                 regressors=[RandomForestRegressor(n_estimators=10),
                                                             RandomForestRegressor(n_estimators=100)],
                                                 select_features=select_features,
                                                 verbose=verbose,
                                                 data_augmentation=data_augmentation)
            self.regressors[ad_type][metric] = model
            print("Model Learned for " + metric + " type: " + ad_type)

            # Storing Model
            model_file = self.get_model_file(ad_type, metric)
            dump(self.regressors[ad_type][metric], model_file)

            # Storing Additional Data
            additional_dict = {
                "model_name": model.__class__.__name__,
                "mae": return_array[0],
                "importances": return_array[2],
                "calculators": len(self.calculators),
                "aggregators": len(self.aggregators)}
            write_dict(additional_dict, model_file.replace(".joblib", "_info.csv"),
                       "additional info for regressor in FRAPPE")

    def predict_metric(self, metric, ad_type, x, y, compute_true=False, verbose=False):

        # Compute True Value
        if compute_true:
            classifiers = frappe_utils.get_supervised_classifiers() \
                if ad_type == "SUP" else frappe_utils.get_unsupervised_classifiers()
            metrics_df, m_scores = \
                frappe_utils.compute_classification_score(dataset_name="dataset_tag",
                                                          x=x, y=y, metrics_df=None,
                                                          classifiers=classifiers)
            true_metric = m_scores[metric]
        else:
            true_metric = None

        # Predict Metric
        start_ms = current_ms()
        ranks_df = self.compute_ranks(dataset_x=x, dataset_y=y, verbose=verbose)
        middle_ms = current_ms()
        model = self.regressors[ad_type][metric]
        data = ranks_df.to_numpy()[0, 1:]
        pred_metric = model.predict(data.reshape(1, -1))[0]

        return pred_metric, middle_ms - start_ms, current_ms() - middle_ms, data, true_metric

    def select_features(self, dataset_x, dataset_y, feature_names, max_drop,
                        ad_type="SUP", metric="mcc", train_split=0.5, verbose=False):

        met_max, t1, t2, data_row, met_calc = self.predict_metric(metric, ad_type, dataset_x, dataset_y,
                                                                  compute_true=True, verbose=False)
        met_threshold = met_max * (1 - max_drop)

        if verbose:
            print("Predicted value of " + str(metric) + " for " + str(ad_type) +
                  " learning using all features is " + str(met_max) + ", calculated is " + str(met_calc))
            print("Selection will stop once pred_" + str(metric) + " goes below " + str(met_threshold))

        # Computing Ranks
        start_time = current_ms()
        ranks_raw = self.compute_raw_dataset_ranks(dataset=dataset_x, label=dataset_y)
        if verbose:
            print("Ranks computed in " + str(current_ms() - start_time) + " seconds")

        # Loading Model
        model = self.regressors[ad_type][metric]

        # Iterating through features
        best_pred = met_threshold
        current_feature_set = numpy.asarray(feature_names)
        feature_sets_list = [{"features": current_feature_set, "pred": met_max, "true": met_calc}]

        while (best_pred >= met_threshold) and len(current_feature_set) > 1:

            if verbose:
                print("Iteration using " + str(len(current_feature_set)) + " features")

            out_list, best_pred, f_to_remove = \
                self.rec_select_features(model, dataset_x, dataset_y,
                                         current_feature_set, ranks_raw,
                                         ad_type, metric, train_split)
            feature_sets_list.extend(out_list)
            if f_to_remove is not None:
                current_feature_set = current_feature_set[current_feature_set != f_to_remove]

        print("Process ended selecting features " + ",".join(f for f in current_feature_set))
        return feature_sets_list

    def rec_select_features(self, model, dataset_x, dataset_y, feature_set, ranks_raw,
                            ad_type="SUP", metric="mcc", train_split=0.5, verbose=True):
        out_list = []
        f_to_remove = None
        best_pred = 0

        for feature in feature_set:

            # Update feature set and ranks
            fs = feature_set[feature_set != feature]
            current_ranks = {}
            for key in ranks_raw:
                current_ranks[key] = ranks_raw[key].drop(labels=[feature])

            # Compute Prediction
            agg_dict, agg_row = self.compute_raw_aggregation(current_ranks)
            pred_metric = model.predict(numpy.asarray(agg_row).reshape(1, -1))[0]
            if best_pred < pred_metric:
                best_pred = pred_metric
                f_to_remove = feature

            # Compute Ground Thruth
            check_clf = DecisionTreeClassifier() if ad_type == "SUP" else COPOD(contamination=0.5)
            truth_metrics = exercise_classifier(dataset_x[fs], dataset_y, check_clf,
                                                train_split=train_split, verbose=False)

            out_list.append({"features": fs, "pred": pred_metric, "true": truth_metrics[metric]})
            if verbose:
                print("pred_" + str(metric) + "=" + str(pred_metric) +
                      ", true_" + str(metric) + "=" + str(truth_metrics[metric]) +
                      " using the feature set of [" + ",".join([str(f) for f in fs]) + "]")

        return out_list, best_pred, f_to_remove
