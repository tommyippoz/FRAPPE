import os
import time
from os.path import isdir


def current_ms():
    """
    Reports the current time in milliseconds
    :return: long int
    """
    return round(time.time() * 1000)


def is_ascending(calc_key):
    if calc_key in {"CosineSimilarity"}:
        return True
    else:
        return False


def get_dataset_files(folder, partial_list=[]):
    for dataset_file in os.listdir(folder):
        full_name = folder + "/" + dataset_file
        if full_name.endswith(".csv"):
            partial_list.append(full_name)
        elif isdir(full_name):
            partial_list = get_dataset_files(full_name, partial_list)
    return partial_list
