
class FrappeAggregator:

    def __init__(self):
        pass

    def calculate_aggregation(self, rank_list, ascending=False):
        return 0.0

    def get_name(self):
        return ""


class GetBest(FrappeAggregator):

    def __init__(self):
        pass

    def calculate_aggregation(self, rank_list, ascending=False):
        if ascending:
            return rank_list.min()
        else:
            return rank_list.max()

    def get_name(self):
        return "GetBest"


class GetSum(FrappeAggregator):

    def __init__(self):
        pass

    def calculate_aggregation(self, rank_list, ascending=False):
        return rank_list.sum()

    def get_name(self):
        return "Sum"


class GetAverage(FrappeAggregator):

    def __init__(self):
        pass

    def calculate_aggregation(self, rank_list, ascending=False):
        return rank_list.mean()

    def get_name(self):
        return "Average"


class GetAverageBest(FrappeAggregator):

    def __init__(self, n):
        self.n = n

    def calculate_aggregation(self, rank_list, ascending=False):
        sorted = rank_list.sort_values(ascending=ascending)
        if self.n-1 < len(sorted):
            subset = sorted.iloc[0:self.n-1]
        else:
            subset = sorted
        return subset.mean()

    def get_name(self):
        return "Average(" + str(self.n) + ")"

