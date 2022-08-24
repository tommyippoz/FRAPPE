import os

from frappe.FrappeInstance import FrappeInstance
from frappe.FrappeType import FrappeType
from utils.dataset_utils import load_binary_tabular_dataset

MODELS_FOLDER = "../models"
DATASET_FILE = "../input_folder/NIDS/ADFANet_Meta.csv"
OUTPUT_FOLDER = "../output_folder"
AD_TYPE = "UNS"
METRIC = "mcc"

if __name__ == '__main__':

    fr_obj = FrappeInstance(load_models=True, instance=FrappeType.REGULAR, models_folder=MODELS_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    [x, y, feature_names, label_names, an_perc, tag] = \
        load_binary_tabular_dataset(DATASET_FILE, label_name="multilabel", limit=1000)
    dataset_name = os.path.basename(DATASET_FILE).replace(".csv", "").split("/")[-1]

    print("\nSelecting features of dataset " + dataset_name)

    # Selecting minimum amount of features
    feature_list, overall_df = fr_obj.select_features(dataset_name=dataset_name,
                                                      dataset_x=x, dataset_y=y,
                                                      feature_names=feature_names,
                                                      max_drop=0.1,
                                                      method="fast",
                                                      ad_type=AD_TYPE, metric=METRIC,
                                                      train_split=0.66, verbose=True)

    with open(OUTPUT_FOLDER + "/select_features_" + dataset_name + ".csv", 'w') as f:
        f.write("pred_met,true_met,features\n")
        for item in feature_list:
            f.write(str(item['pred']) + "," + str(item['true']) + "," +
                    ";".join(str(f) for f in item['features']) + "\n")
