import copy

import numpy
import pandas
import scipy
import sklearn
import sklearn.feature_selection as fs
import skrebate
from genetic_selection import GeneticSelectionCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from skrebate import SURF


class FrappeRanker:
    """
    Abstract class for Rankers in FRAPPE
    """

    def __init__(self):
        pass

    def compute_rank(self, dataset, label):
        """
        Computes the Rank w.r.t. the label for each feature in the dataset
        :param dataset: the input dataset containing features
        :param label: label to calculate rank against
        :return: double value
        """
        pass

    def get_ranker_name(self):
        """
        Returns the name of the ranker (string)
        """
        pass


class FrappeSingleRanker(FrappeRanker):
    """
    Abstract class for Rankers in FRAPPE that calculate their values looking at each feature individually
    """

    def compute_rank(self, dataset, label):
        features = dataset.columns
        rank = numpy.zeros(len(features))
        for i in range(0, len(features)):
            feat_array = numpy.asarray(dataset[features[i]])
            if not numpy.all(feat_array == feat_array[0]):
                rank[i] = self.compute_single_rank(numpy.asarray(dataset[features[i]]), label)
            else:
                rank[i] = 0.0
        return rank

    def compute_single_rank(self, feature, label):
        """
        Computes the Rank w.r.t. the label for a specific feature in the dataset
        :param feature: the input feature from dataset
        :param label: label to calculate rank against
        :return: double value
        """
        return 0.0


class RSquaredRanker(FrappeSingleRanker):
    """
    Ranker using the R-Squared Statistical Index from SciPy
    """

    def compute_single_rank(self, feature, label):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(feature, label)
        return r_value ** 2

    def get_ranker_name(self):
        return "R-Squared"


class PearsonRanker(FrappeSingleRanker):
    """
    Ranker using the Pearson Correlation Index from SciPy
    """

    def compute_single_rank(self, feature, label):
        return scipy.stats.pearsonr(feature, label)[0]

    def get_ranker_name(self):
        return "Pearson"


class CosineSimilarityRanker(FrappeSingleRanker):
    """
    Ranker using the Cosine Distance from SciPy
    """

    def compute_single_rank(self, feature, label):
        return 1 - scipy.spatial.distance.cosine(feature, label)

    def get_ranker_name(self):
        return "CosineSimilarity"


class SpearmanRanker(FrappeSingleRanker):
    """
    Ranker using the Spearman Statistical Index from SciPy
    """

    def compute_single_rank(self, feature, label):
        return scipy.stats.spearmanr(feature, label)[0]

    def get_ranker_name(self):
        return "Spearman"


class ChiSquaredRanker(FrappeSingleRanker):
    """
    Ranker using the Chi-Squared Statistical Index from Scikit-Learn
    """

    def compute_single_rank(self, feature, label):
        scaled_feat_values = sklearn.preprocessing.MinMaxScaler().fit_transform(feature.reshape(-1, 1))
        return fs.chi2(scaled_feat_values, label)[0][0]

    def get_ranker_name(self):
        return "ChiSquared"


class MutualInfoRanker(FrappeSingleRanker):
    """
    Ranker using the Mutual Information Index from Scikit-Learn
    """

    def compute_single_rank(self, feature, label):
        return sklearn.feature_selection.mutual_info_classif(feature.reshape(-1, 1), label)[0]

    def get_ranker_name(self):
        return "MutualInfo"


class ANOVARanker(FrappeSingleRanker):
    """
    Ranker using the ANOVA R from Scikit-Learn
    """

    def compute_single_rank(self, feature, label):
        return sklearn.feature_selection.f_classif(feature.reshape(-1, 1), label)[0]

    def get_ranker_name(self):
        return "ANOVAF"


class ReliefRanker(FrappeRanker):
    """
    Ranker using the Mutual Information Index from Scikit-Learn
    """

    def __init__(self, limit_rows=1000):
        FrappeRanker.__init__(self)
        self.limit_rows = limit_rows

    def compute_rank(self, dataset, label):
        ranker = self.init_ranker(len(dataset.columns))
        x = dataset.to_numpy()
        if len(dataset.index) > self.limit_rows:
            index = numpy.where(label == 1)[0][0]
            if index + self.limit_rows < len(label):
                x = x[index:index + self.limit_rows]
                label = label[index:index + self.limit_rows]
            else:
                x = x[index - self.limit_rows:index]
                label = label[index - self.limit_rows:index]
        ranker.fit(x, label)
        return ranker.feature_importances_

    def init_ranker(self, n_features):
        return


class SURFRanker(ReliefRanker):

    def __init__(self, limit_rows=1000):
        ReliefRanker.__init__(self, limit_rows)

    def init_ranker(self, n_features):
        return SURF(n_features_to_select=n_features, n_jobs=-1)

    def get_ranker_name(self):
        return "SURF"


class ReliefFRanker(ReliefRanker):

    def __init__(self, limit_rows=1000, n_neighbours=20):
        ReliefRanker.__init__(self, limit_rows)
        self.n_neigbours = n_neighbours

    def init_ranker(self, n_features):
        return skrebate.ReliefF(n_neighbors=self.n_neigbours, n_features_to_select=n_features, n_jobs=-1)

    def get_ranker_name(self):
        return "ReliefF(" + str(self.n_neigbours) + ")"


class SURFStarRanker(ReliefRanker):

    def __init__(self, limit_rows=1000):
        ReliefRanker.__init__(self, limit_rows)

    def init_ranker(self, n_features):
        return skrebate.SURFstar(n_features_to_select=n_features, n_jobs=-1)

    def get_ranker_name(self):
        return "SURF*"


class MultiSURFRanker(ReliefRanker):

    def __init__(self, limit_rows=1000):
        ReliefRanker.__init__(self, limit_rows)

    def init_ranker(self, n_features):
        return skrebate.MultiSURF(n_features_to_select=n_features, n_jobs=-1)

    def get_ranker_name(self):
        return "MultiSURF"


class WrapperRanker(FrappeRanker):
    """
    Ranker that wraps classifiers and gets feature_importance
    """

    def __init__(self, classifier, as_pandas=False):
        FrappeRanker.__init__(self)
        self.classifier = classifier
        self.as_pandas = as_pandas

    def compute_rank(self, dataset, label):
        """
        Computes the Rank w.r.t. the label for each feature in the dataset
        :param dataset: the input dataset containing features
        :param label: label to calculate rank against
        :return: double value
        """
        if (not self.as_pandas) & (isinstance(dataset, pandas.DataFrame)):
            dataset = dataset.to_numpy()
        x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(dataset, label, test_size=0.9)
        temp_classifier = copy.deepcopy(self.classifier)
        temp_classifier.fit(x_tr, y_tr)
        return temp_classifier.feature_importances_

    def get_ranker_name(self):
        """
        Returns the name of the ranker (string)
        """
        return "Wrap(" + str(self.classifier.__class__.__name__) + ")"


class GeneticRanker(FrappeRanker):
    """
    Ranker that calculates rank using genetic algorithm
    """

    def __init__(self, as_pandas=False):
        FrappeRanker.__init__(self)
        self.as_pandas = as_pandas

    def compute_rank(self, dataset, label):
        """
        Computes the Rank w.r.t. the label for each feature in the dataset
        :param dataset: the input dataset containing features
        :param label: label to calculate rank against
        :return: double value
        """
        if (not self.as_pandas) & (isinstance(dataset, pandas.DataFrame)):
            dataset = dataset.to_numpy()

        estimators = RandomForestClassifier()
        selectors = GeneticSelectionCV(estimators, scoring='accuracy', n_population=2, n_generations=2, n_jobs=1)
        selectors = selectors.fit(dataset, label)

        a = selectors.support_
        return a

    def get_ranker_name(self):
        """
        Returns the name of the ranker (string)
        """
        return "GeneticAlgorithm"

class CoefRanker(FrappeRanker):
    """
    Ranker that wraps classifiers and gets feature_importance
    """

    def __init__(self, classifier, as_pandas=False):
        FrappeRanker.__init__(self)
        self.classifier = classifier
        self.as_pandas = as_pandas

    def compute_rank(self, dataset, label):
        """
        Computes the Rank w.r.t. the label for each feature in the dataset
        :param dataset: the input dataset containing features
        :param label: label to calculate rank against
        :return: double value
        """
        if (not self.as_pandas) & (isinstance(dataset, pandas.DataFrame)):
            dataset = dataset.to_numpy()
        x_tr, x_te, y_tr, y_te = sklearn.model_selection.train_test_split(dataset, label, test_size=0.5)
        temp_classifier = copy.deepcopy(self.classifier)
        temp_classifier.fit(x_tr, y_tr)
        return temp_classifier.coef_

    def get_ranker_name(self):
        """
        Returns the name of the ranker (string)
        """
        return "Wrap(" + str(self.classifier.__class__.__name__) + ")"

