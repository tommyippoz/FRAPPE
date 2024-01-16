import os

from frappe.FrappeInstance import FrappeInstance
from tqdm import tqdm
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset

MODELS_FOLDER = "models"
TEST_INPUT_FOLDER = "novel_datasets_featureselection"
OUTPUT_FOLDER = "output_folder"
OUT_FILE = "datasets_prediction_sup.csv"
AD_TYPE = "SUP"

if __name__ == '__main__':

    with open(OUTPUT_FOLDER + "/" + OUT_FILE, 'w') as f:
        f.write("dataset_tag,metric,predicted,time_fd,time_reg,data_row\n")

    fr_obj = FrappeInstance(load_models=True, rankers_set="reduced")

    dataset_files = frappe_utils.get_dataset_files(TEST_INPUT_FOLDER)

    for metric in ["mcc", "auc"]:

        for file_name in tqdm(dataset_files, desc="Datasets Progress"):
            print("\n------------------------------------------------------\n")

            [x_data, y_data, feature_names, label_names, an_perc, tag] = \
                load_binary_tabular_dataset(file_name, label_name="binlabel", normal_tag=0)[0]
            dataset_name = os.path.basename(file_name).replace(".csv", "")

            print("\nPredicting tag " + tag + " of dataset " + dataset_name)

            dataset_tag = dataset_name + "@" + tag

            pred_met, feature_data_time, reg_time, data_row, metrics = fr_obj.predict_metric(metric, AD_TYPE, x_data,
                                                                                             y_data)

            print("[" + dataset_tag + "] Predicted " + metric + ": " + str(pred_met) +
                  " time: [" + str(feature_data_time) + "; " + str(reg_time) + "]")

            with open(OUTPUT_FOLDER + "/" + OUT_FILE, 'a') as f:
                # create the csv writer
                f.write(dataset_tag + "," + metric + "," + str(pred_met) + "," +
                        str(feature_data_time) + "," + str(reg_time) + "," +
                        str(len(y_data)) + "," + str(x_data.shape[1]) + ","
                                                                        ','.join(map(str, data_row)) + "\n")
