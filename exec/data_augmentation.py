import os

import numpy

from frappe.FrappeInstance import FrappeInstance
from frappe.FrappeType import FrappeType
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset

MODELS_FOLDER = "../models"
DATASET_FOLDER = "../input_folder/NIDS/"
OUTPUT_FOLDER = "../output_folder"
AD_TYPE = "SUP"
METRIC = "mcc"

if __name__ == '__main__':

    fr_obj = FrappeInstance(load_models=True, instance=FrappeType.REGULAR, models_folder=MODELS_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    dataset_files = frappe_utils.get_dataset_files(DATASET_FOLDER)
    dataaug_file = OUTPUT_FOLDER + "/" + AD_TYPE + "_dataaug_ids.csv"

    for dataset_file in dataset_files:

        [x, y, feature_names, label_names, an_perc, tag] = \
            load_binary_tabular_dataset(dataset_file, label_name="multilabel", limit=100000)
        dataset_name = os.path.basename(dataset_file).replace(".csv", "").split("/")[-1]

        print("Selecting features of dataset " + dataset_name)

        # Selecting minimum amount of features
        overall_df, feature_list = fr_obj.select_features(dataset_name=dataset_name,
                                                          dataset_x=x, dataset_y=y,
                                                          feature_names=feature_names,
                                                          max_drop=0.4,
                                                          compute_true=True,
                                                          ad_type=AD_TYPE, metric=METRIC,
                                                          train_split=0.66, verbose=True)

        if not os.path.exists(dataaug_file):
            overall_df.to_csv(dataaug_file, index=False)
        else:
            overall_df.to_csv(dataaug_file, index=False, mode='a', header=False)
