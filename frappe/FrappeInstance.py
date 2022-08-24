import hashlib
import os

import numpy
import pandas
from pyod.models.copod import COPOD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
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
        for calculator in tqdm(self.calculators, "Calculating Feature Rankings, it may take a while..."):
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

    def predict_metric(self, metric, ad_type, x, y, dataset_name=None,
                       compute_true="no", agg_ranks=None, verbose=False):

        # Compute True Value
        if compute_true is not "no":
            if compute_true is "full":
                classifiers = frappe_utils.get_supervised_classifiers() \
                    if ad_type == "SUP" else frappe_utils.get_unsupervised_classifiers(outliers_fraction=0.5)
            else:
                classifiers = frappe_utils.get_fast_supervised_classifiers() \
                    if ad_type == "SUP" else frappe_utils.get_fast_unsupervised_classifiers(outliers_fraction=0.5)
            metrics_df, m_scores = \
                frappe_utils.compute_classification_score(
                    dataset_name="dataset_tag" if dataset_name is None else dataset_name,
                    x=x, y=y, metrics_df=None, classifiers=classifiers, verbose=verbose)
        else:
            m_scores = None

        # Compute Ranks
        start_ms = current_ms()
        if agg_ranks is None:
            ranks_df = self.compute_ranks(dataset_x=x, dataset_y=y, verbose=verbose)
            data = ranks_df.to_numpy()[0, 1:]
        else:
            data = numpy.asarray(agg_ranks)
        middle_ms = current_ms()

        # Predict using the Model
        model = self.regressors[ad_type][metric]
        pred_metric = model.predict(data.reshape(1, -1))[0]

        return pred_metric, middle_ms - start_ms, current_ms() - middle_ms, data, m_scores

    def select_features(self, dataset_name, dataset_x, dataset_y, feature_names, max_drop,
                        ad_type="SUP", metric="mcc", compute_true=False, train_split=0.5, verbose=False):

        # Computing Ranks
        start_time = current_ms()
        ranks_raw = self.compute_raw_dataset_ranks(dataset=dataset_x, label=dataset_y)
        if verbose:
            print("Ranks computed in " + str(current_ms() - start_time) + " seconds")

        # Computing True / Pred values using all features (maximum classification performance)
        agg_dict, agg_row = self.compute_raw_aggregation(ranks_raw)
        pred_max, t1, t2, data_row, calc_metrics = self.predict_metric(metric, ad_type, dataset_x, dataset_y,
                                                                       compute_true="fast" if compute_true else "no",
                                                                       agg_ranks=agg_row, verbose=False)
        met_threshold = pred_max * (1 - max_drop)
        calc_max = calc_metrics[metric] if compute_true else None
        if verbose:
            print("Pred_" + str(metric) + "@" + str(ad_type) + " = " + str(pred_max))
            if compute_true:
                print("Calculated metric value is " + str(calc_max))
            print("Selection will stop once pred_" + str(metric) + "@" + str(ad_type) + " < " + str(met_threshold))

        # Iterating through features
        overall_df = None
        pred_met = met_threshold
        current_feature_set = numpy.asarray(feature_names)

        while (pred_met >= met_threshold) and len(current_feature_set) > 1:

            if verbose:
                print("Iteration using " + str(len(current_feature_set)) + " features")

            selection_df = None
            for feature in tqdm(current_feature_set, "Current Feature Set"):
                # Computing prediction and updating results
                selection_df, pm = self.analyze_feature_set(dataset_name, metric, ad_type, dataset_x, dataset_y,
                                                            [feature], current_feature_set, ranks_raw,
                                                            compute_true=compute_true,
                                                            selection_df=selection_df, verbose=False)

            selection_df.sort_values(by=["pred_met"], ascending=True, inplace=True)

            remove_df = None
            n_features_to_remove = None
            iter_flag = True
            while (n_features_to_remove is None or n_features_to_remove > 0) and iter_flag:
                n_features_to_remove = int(len(current_feature_set) / 2) if n_features_to_remove is None \
                    else int(n_features_to_remove / 2)
                features_to_remove = selection_df["to_remove"][-n_features_to_remove:].tolist()
                remove_df, pred_met = self.analyze_feature_set(dataset_name, metric, ad_type, dataset_x, dataset_y,
                                                               features_to_remove, current_feature_set, ranks_raw,
                                                               compute_true=compute_true,
                                                               selection_df=remove_df, verbose=False)
                if pred_met >= met_threshold:
                    iter_flag = False

            if n_features_to_remove > 0:
                for f_remove in features_to_remove:
                    current_feature_set = current_feature_set[current_feature_set != f_remove]
            if verbose:
                if n_features_to_remove > 0:
                    print("Removing Features: [" + ", ".join(features_to_remove) + "]" +
                          " pred_" + metric + "@" + ad_type + " = " + str(pred_met))
                else:
                    print("Cannot remove features without dropping performance, process is going to stop")
                    break

            if overall_df is None:
                overall_df = pandas.concat([selection_df, remove_df])
            else:
                overall_df = pandas.concat([overall_df, selection_df, remove_df])

        print("Selection process ended selecting " + str(len(current_feature_set)) + "/" + str(len(dataset_x.columns)) +
              " features: [" + ",".join(f for f in current_feature_set) + "]")
        print("pred_" + metric + "@" + ad_type + " becomes " + str(pred_met) + " instead of " + str(pred_max))
        return overall_df, current_feature_set

    def analyze_feature_set(self, dataset_name, metric, ad_type, dataset_x, dataset_y,
                            features_to_remove, current_feature_set, ranks_raw,
                            compute_true=False, selection_df=None, verbose=False):
        # Update feature set and ranks
        fs = current_feature_set
        for f_remove in features_to_remove:
            fs = fs[fs != f_remove]
        current_ranks = {}
        for key in ranks_raw:
            current_ranks[key] = ranks_raw[key].drop(labels=features_to_remove)

        # Computing Prediction
        full_name = dataset_name + "_" + str(len(fs)) + "_" + hashlib.md5(";".join(fs).encode('utf-8')).hexdigest()
        agg_dict, agg_row = self.compute_raw_aggregation(current_ranks)
        pred_met, t1, t2, data_row, calc_metrics = self.predict_metric(metric, ad_type, dataset_x[fs], dataset_y,
                                                                       dataset_name=full_name,
                                                                       compute_true="fast" if compute_true else "no",
                                                                       agg_ranks=agg_row, verbose=verbose)
        # Writing output
        selection_df = self.update_selection_df(pred_met, full_name, features_to_remove, fs, calc_metrics, agg_dict,
                                                selection_df)
        return selection_df, pred_met

    def update_selection_df(self, pred_met, dataset_name, to_remove, feature_set, truth_metrics, agg_ranks,
                            selection_df=None):

        if selection_df is None:
            # Create new Dataframe selection_df
            tag_list = ["dataset_name", "features", "n_features", "to_remove", "pred_met"] + \
                       [str(j) + "_" + str(i) for j in agg_ranks for i in agg_ranks[j]]
            if truth_metrics is not None:
                tag_list = tag_list + [k for k in truth_metrics]
            selection_df = pandas.DataFrame(columns=tag_list)

        # Create new entry in selection_df
        selection_df.loc[len(selection_df), "dataset_name"] = dataset_name
        selection_df.loc[selection_df.dataset_name == dataset_name, "features"] = ";".join(feature_set)
        selection_df.loc[selection_df.dataset_name == dataset_name, "n_features"] = len(feature_set)
        selection_df.loc[selection_df.dataset_name == dataset_name, "to_remove"] = ";".join(to_remove)
        selection_df.loc[selection_df.dataset_name == dataset_name, "pred_met"] = pred_met
        for ranker in agg_ranks:
            for agg in agg_ranks[ranker]:
                selection_df.loc[selection_df.dataset_name == dataset_name, ranker + "_" + agg] = agg_ranks[ranker][agg]
        if truth_metrics is not None:
            for metric in truth_metrics:
                selection_df.loc[selection_df.dataset_name == dataset_name, metric] = truth_metrics[metric]

        return selection_df
