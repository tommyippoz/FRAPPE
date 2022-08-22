import os

import numpy

from frappe.FrappeInstance import FrappeInstance
from frappe.FrappeType import FrappeType
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset

MODELS_FOLDER = "../models"
DATASET_FOLDER = "../input_folder/Biometry_Datasets/"
OUTPUT_FOLDER = "../output_folder"
AD_TYPE = "UNS"
METRIC = "mcc"

if __name__ == '__main__':

    fr_obj = FrappeInstance(load_models=True, instance=FrappeType.REGULAR, models_folder=MODELS_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    dataset_files = frappe_utils.get_dataset_files(DATASET_FOLDER)

    for dataset_file in dataset_files:

        [x, y, feature_names, label_names, an_perc, tag] = \
            load_binary_tabular_dataset(dataset_file, label_name="multilabel", limit=numpy.NaN)
        dataset_name = os.path.basename(dataset_file).replace(".csv", "").split("/")[-1]

        print("\nSelecting features of dataset " + dataset_name)

        # Selecting minimum amount of features
        feature_list, overall_df = fr_obj.select_features(dataset_name=dataset_name,
                                                          dataset_x=x, dataset_y=y,
                                                          feature_names=feature_names,
                                                          max_drop=0.1,
                                                          ad_type=AD_TYPE, metric=METRIC,
                                                          train_split=0.66, verbose=True)

        overall_df.to_csv(OUTPUT_FOLDER + "/" + AD_TYPE + "_dataaug.csv", index=False)
