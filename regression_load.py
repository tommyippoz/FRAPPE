import csv

import pandas
import sklearn
from tqdm import tqdm

from frappe.FrappeInstance import FrappeInstance
import matplotlib.pyplot as plt

INPUT_FILES = [["SUP", "input_regression/20220309_SupervisedCorrelation_Wrap50.csv"]]
OUTPUT_FOLDER = "output_folder"

METRIC_NAMES = ["auc"]

OUTPUT_FLAG = True

if __name__ == '__main__':

    for [desc, file] in INPUT_FILES:

        # Load File
        df = pandas.read_csv(file, sep=",")

        for metric in METRIC_NAMES:

            best = 0
            iterator = tqdm(range(0, 10), desc="Iterating for metric " + metric + " on file " + desc)
            for i in iterator:

                df = df.sample(frac=1.0)
                df = df.fillna(0)
                df = df.replace('null', 0)

                frappe = FrappeInstance()
                frappe.load_dataframe(df)

                try:
                    [metric_scores, correl, x_te, y_te, y_pred] = frappe.regression_analysis(target_metric=metric,
                                                                                             train_split=0.8,
                                                                                             select_features=None,
                                                                                             data_augmentation=True,
                                                                                             verbose=False)
                    if (best == 0) or (metric_scores["mean_abs_err"] < best[0]["mean_abs_err"]):
                        best = [metric_scores, correl, x_te, y_te, y_pred]

                except:
                    print("err")

            iterator.close()
            print("\nBest regressor gets Mean Absolute Error of " + str(best[0]['mean_abs_err'])
                  + " and R2 of " + str(best[0]['r2']))

            if OUTPUT_FLAG:
                x_te = best[2]
                x_te["true_value"] = best[3]
                x_te["predicted_value"] = best[4]
                x_te.to_csv(OUTPUT_FOLDER + "/" + desc + "_" + metric + ".csv", index=False)

                with open(OUTPUT_FOLDER + "/corr_" + desc + "_" + metric + ".csv", 'w') as f:
                    w = csv.DictWriter(f, best[1].keys())
                    w.writeheader()
                    w.writerow(best[1])

        # Plot
        # plt.scatter(y_te, y_pred)
        # plt.show()


