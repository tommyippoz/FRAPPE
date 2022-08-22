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


def build_ranks_scores():
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

    ranks_df.to_csv(OUTPUT_FOLDER + "/df_ranks.csv", index=False)

    # Compute Metrics
    sup_metrics_df = None
    for file_name in tqdm(dataset_files, desc="Classification Scores - Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset_array_partition(file_name, label_name="multilabel",
                                                                    n_partitions=5)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nRanking tag " + tag + " of dataset " + dataset_name)

            dataset_tag = dataset_name + "@" + tag

            try:

                sup_metrics_df, m_scores = frappe_utils.compute_classification_score(dataset_name=dataset_tag,
                                                                                     x=x,
                                                                                     y=y,
                                                                                     metrics_df=sup_metrics_df,
                                                                                     classifiers=frappe_utils.get_supervised_classifiers())
            except:
                print("Cannot compute classification metrics")

    sup_metrics_df.to_csv(OUTPUT_FOLDER + "/df_sup_metrics.csv", index=False)

    # Compute Metrics
    uns_metrics_df = None
    for file_name in tqdm(dataset_files, desc="Classification Scores - Datasets Progress"):

        print("\n------------------------------------------------------\n")

        dataset_array = load_binary_tabular_dataset_array_partition(file_name, label_name="multilabel",
                                                                    n_partitions=5)
        dataset_name = os.path.basename(file_name).replace(".csv", "")

        for [x, y, feature_names, label_names, an_perc, tag] in tqdm(dataset_array, desc="Variants Progress"):
            print("\nRanking tag " + tag + " of dataset " + dataset_name)

            dataset_tag = dataset_name + "@" + tag

            try:

                uns_metrics_df, m_scores = frappe_utils.compute_classification_score(dataset_name=dataset_tag,
                                                                                     x=x,
                                                                                     y=y,
                                                                                     metrics_df=uns_metrics_df,
                                                                                     classifiers=frappe_utils.get_unsupervised_classifiers(
                                                                                         outliers_fraction=an_perc))
            except:
                print("Cannot compute classification metrics")

    uns_metrics_df.to_csv(OUTPUT_FOLDER + "/df_uns_metrics.csv", index=False)

    sup_df = pandas.concat([ranks_df, sup_metrics_df], axis=1, join="inner")
    sup_df.to_csv(OUTPUT_FOLDER + "/sup_ranks.csv", index=False)

    uns_df = pandas.concat([ranks_df, uns_metrics_df], axis=1, join="inner")
    uns_df.to_csv(OUTPUT_FOLDER + "/uns_ranks.csv", index=False)

    return sup_df, uns_df


if __name__ == '__main__':

    fr_obj = FrappeInstance(load_models=False, instance=FrappeType.REGULAR, models_folder=MODELS_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    # Supervised Ranks
    sup_ranks = None
    sup_ranks_file = OUTPUT_FOLDER + "/sup_ranks.csv"
    if os.path.exists(sup_ranks_file):
        sup_ranks = pandas.read_csv(sup_ranks_file)

    # Unsupervised Ranks
    uns_ranks = None
    uns_ranks_file = OUTPUT_FOLDER + "/uns_ranks.csv"
    if os.path.exists(uns_ranks_file):
        uns_ranks = pandas.read_csv(uns_ranks_file)

    if uns_ranks is None or sup_ranks is None:
        sup_ranks, uns_ranks = build_ranks_scores()

    # Learning Models for SUPERVISED
    fr_obj.learn_models(sup_ranks, ad_type="SUP", train_split=0.8,
                        select_features=None, data_augmentation=False, verbose=True)

    # Learning Models for UNSUPERVISED
    fr_obj.learn_models(uns_ranks, ad_type="UNS", train_split=0.8,
                        select_features=None, data_augmentation=False, verbose=True)
