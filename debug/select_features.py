import os

from frappe.FrappeInstance import FrappeInstance
from frappe.FrappeType import FrappeType
from utils.dataset_utils import load_binary_tabular_dataset

MODELS_FOLDER = "../models"
DATASET_FILE = "../input/test/weatherAUS_pre.csv"
OUTPUT_FOLDER = "../output_folder"
AD_TYPE = "UNS"
METRIC = "mcc"

if __name__ == '__main__':

    fr_obj = FrappeInstance(load_models=True, instance=FrappeType.REGULAR, models_folder=MODELS_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    [x, y, feature_names, label_names, an_perc, tag] = \
        load_binary_tabular_dataset(DATASET_FILE, normal_tag=0, label_name="multilabel", limit=200000)
    dataset_name = os.path.basename(DATASET_FILE).replace(".csv", "").split("/")[-1]

    print("\nSelecting features of dataset " + dataset_name)

    # Selecting minimum amount of features
    overall_df, feature_list = fr_obj.select_features(dataset_name=dataset_name,
                                                      dataset_x=x, dataset_y=y,
                                                      feature_names=feature_names,
                                                      max_drop=0.05,
                                                      compute_true="no",
                                                      ad_type=AD_TYPE, metric=METRIC,
                                                      train_split=0.66, verbose=True)

    overall_df.to_csv(OUTPUT_FOLDER + "/" + dataset_name + "selection_" + AD_TYPE + ".csv", index=False)
