import os

import pandas
from frappe.FrappeInstance import FrappeInstance
from tqdm import tqdm
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset_array_partition

MODELS_FOLDER = "models"
TEST_INPUT_FOLDER = "novel_datasets_featureselection"
OUTPUT_FOLDER = "output_folder"

TRUTH_FILES = [["SUP", "input_reg/20220309_SupervisedCorrelation_Wrap50.csv"]]  # ,
# ["UNSUP", "input_reg/20220309_UnsupervisedCorrelation_Wrap50.csv"]]


if __name__ == '__main__':

    with open(OUTPUT_FOLDER + "/timings_prediction.csv", 'w') as f:
        f.write("dataset_tag,desc,metric,predicted,true,mae,time_fd,time_reg,data_row\n")

    frappe = FrappeInstance(load_models=True)

    dataset_files = frappe_utils.get_dataset_files(TEST_INPUT_FOLDER)

    for metric in ["mcc", "auc"]:

        for desc, path in TRUTH_FILES:

            ground_truth = pandas.read_csv(path, sep=",")

            for file_name in tqdm(dataset_files, desc="Datasets Progress"):

                print("\n------------------------------------------------------\n")

                dataset_array = load_binary_tabular_dataset_array_partition(file_name, label_name="multilabel",
                                                                            n_partitions=5)
                dataset_name = os.path.basename(file_name).replace(".csv", "")

                for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
                    print("\nPredicting tag " + tag + " of dataset " + dataset_name)

                    dataset_tag = dataset_name + "@" + tag

                    pred_met, feature_data_time, reg_time, data_row, metrics = frappe.predict_metric(metric, x, y)

                    true_met = ground_truth.loc[ground_truth['dataset_name'] == dataset_tag][metric].iloc[0]

                    print("[" + desc + "] Predicted " + metric + ": " + str(pred_met) + " was " + str(true_met)
                          + " mae of " + str(abs(pred_met - true_met)) +
                          " time: [" + str(feature_data_time) + "; " + str(reg_time) + "]")

                    with open(OUTPUT_FOLDER + "/timings_prediction.csv", 'a') as f:
                        # create the csv writer
                        f.write(dataset_tag + "," + desc + "," + metric + "," + str(pred_met) + "," +
                                str(true_met) + "," + str(abs(pred_met - true_met)) + "," +
                                str(feature_data_time) + "," + str(reg_time) + "," +
                                str(len(y)) + "," + str(x.shape[1]) + ","
                                                                      ','.join(map(str, data_row)) + "\n")
