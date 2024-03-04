import os

import numpy
from joblib import load
from tqdm import tqdm

from src import frappe_utils
from src.FrappeInstance import FrappeInstance
from src.FrappeType import FrappeType
from src.dataset_utils import load_tabular_dataset, \
    load_binary_tabular_dataset

MODELS_FOLDER = "../models"
TEST_INPUT_FOLDER = "../input/test"
OUTPUT_FOLDER = "../output_folder"


if __name__ == '__main__':

    with open(OUTPUT_FOLDER + "/prediction_tests_fast.csv", 'w') as f:
        f.write("dataset_tag,task,metric,predicted,true,mae,time_fd,time_reg,test_size,n_features,n_classes,data_row\n")

    dataset_files = frappe_utils.get_dataset_files(TEST_INPUT_FOLDER)

    for file_name in tqdm(dataset_files, desc="Datasets Progress"):

        dataset_name = os.path.basename(file_name).replace(".csv", "")
        print("\n------------------------------------------------------\n")
        print(dataset_name + "\n")

        for task in ["bin-sup", "bin-uns", "multi"]:

            for metric in ["mcc", "auc"]:

                if "bin-" in task:
                    [x, y, labels, feature_list, an_perc, tag] = \
                        load_binary_tabular_dataset(file_name, label_name="multilabel")
                else:
                    x, y, labels, feature_list = load_tabular_dataset(file_name, label_name="multilabel")

                fr_obj = FrappeInstance(classification_type=task, target_metric=metric,
                                        instance=FrappeType.FAST, models_folder=MODELS_FOLDER)

                pred_met, feature_data_time, reg_time, data_row, true_mets = \
                    fr_obj.predict_metric(x, y, dataset_name, compute_true=True)

                print("[" + task + "@" + metric + "] Predicted: " + str(pred_met) + " was " + str(true_mets[metric])
                      + " ae of " + str(abs(pred_met - true_mets[metric])) +
                      " time: [" + str(feature_data_time) + "; " + str(reg_time) + "]")

                with open(OUTPUT_FOLDER + "/prediction_tests_fast.csv", 'a') as f:
                    # create the csv writer
                    f.write(dataset_name + "," + task + "," + metric + "," + str(pred_met) + "," +
                            str(true_mets[metric]) + "," + str(abs(pred_met - true_mets[metric])) + "," +
                            str(feature_data_time) + "," + str(reg_time) + "," +
                            str(len(y)) + "," + str(x.shape[1]) + "," + str(len(numpy.unique(labels))) + "," +
                            ','.join(map(str, data_row)) + "\n")
