import os
from collections import Counter

import numpy


def make_directory(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def detect_outliers(dataframe, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        q1 = numpy.percentile(dataframe[col], 25)
        # 3rd quartile (75%)
        q3 = numpy.percentile(dataframe[col], 75)
        # Interquartile range (IQR)
        iqr = q3 - q1

        # outlier step
        outlier_step = 1.5 * iqr

        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataframe[(dataframe[col] < q1 - outlier_step) | (dataframe[col] > q3 + outlier_step)].index

        # append the found outlier indices for col
        # to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers
