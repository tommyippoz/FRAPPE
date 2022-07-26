import os
import pandas
from tqdm import tqdm

from frappe.FrappeInstance import FrappeInstance
from frappe.FrappeType import FrappeType
from utils import frappe_utils
from utils.dataset_utils import load_binary_tabular_dataset_array_partition

MODELS_FOLDER = "../models"
INPUT_FOLDER = "../input_folder"
OUTPUT_FOLDER = "../output_folder"

if __name__ == '__main__':

    fr_obj = FrappeInstance(load_models=False, instance=FrappeType.REGULAR, models_folder=MODELS_FOLDER)

    dataset_files = frappe_utils.get_dataset_files(INPUT_FOLDER)

    # Compute Rankings
    ranks_df = None
    for file_name in tqdm(dataset_files, desc="Feature Ranking - Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset_array_partition(file_name, label_name="multilabel",
                                                                    n_partitions=5)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nRanking tag " + tag + " of dataset " + dataset_name)

            dataset_tag = dataset_name + "@" + tag

            ranks_df = fr_obj.compute_ranks(dataset_name=dataset_tag,
                                            dataset_x=x,
                                            dataset_y=y,
                                            ranks_df=ranks_df)

    ranks_df.to_csv("df_ranks.csv", index=False)

    # Compute Metrics
    metrics_df = None
    for file_name in tqdm(dataset_files, desc="Classification Scores - Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset_array_partition(file_name, label_name="multilabel",
                                                                    n_partitions=5)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nRanking tag " + tag + " of dataset " + dataset_name)

            dataset_tag = dataset_name + "@" + tag

            try:

                metrics_df, m_scores = frappe_utils.compute_classification_score(dataset_name=dataset_tag,
                                                                                 x=x,
                                                                                 y=y,
                                                                                 metrics_df=metrics_df,
                                                                                 classifiers=frappe_utils.get_supervised_classifiers())
            except:
                print("Cannot compute classification metrics")

    metrics_df.to_csv("df_sup_metrics.csv", index=False)

    # Compute Metrics
    metrics_df = None
    for file_name in tqdm(dataset_files, desc="Classification Scores - Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset_array_partition(file_name, label_name="multilabel",
                                                                    n_partitions=5)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nRanking tag " + tag + " of dataset " + dataset_name)

            dataset_tag = dataset_name + "@" + tag

            try:

                metrics_df, m_scores = frappe_utils.compute_classification_score(dataset_name=dataset_tag,
                                                                                 x=x,
                                                                                 y=y,
                                                                                 metrics_df=metrics_df,
                                                                                 classifiers=frappe_utils.get_unsupervised_classifiers(outliers_fraction=an_perc))
            except:
                print("Cannot compute classification metrics")

    metrics_df.to_csv("df_uns_metrics.csv", index=False)
