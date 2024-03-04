# FRAPPE

Python library that exercises Feature RAnkers to Predict classification PerformancE of binary classifiers

More information on the project can be found in the submitted paper. Below you find a resume and an usage example.

## Abstract

Machine Learning algorithms that perform classification are increasingly been adopted in Information and Communication Technology (ICT) systems and infrastructures due to their capability to profile their expected behavior and detect anomalies due to ongoing errors or intrusions. Deploying a classifier for a given system requires conducting comparison and sensitivity analyses that are time-consuming, require domain expertise, and may even not achieve satisfactory classification performance, resulting in a waste of money and time for practitioners and stakeholders. This paper predicts the expected performance of classifiers without needing to select, craft, exercise, and compare them, requiring minimal expertise and machinery. Should classification performance be predicted worse than expectations, the users could focus on improving data quality and monitoring systems instead of wasting time in exercising classifiers, saving key time and money. The prediction strategy uses scores of feature rankers, which are processed by regressors to predict metrics as Matthews Correlation Coefficient (MCC) and Area Under roc-Curve (AUC) for quantifying classification performance. We validate our prediction strategy through a massive experimental analysis using up to 12 feature rankers that process features from 23 public datasets, creating additional variants in the process and exercising supervised and unsupervised classifiers. Our findings show that it is possible to predict the value of performance metrics for supervised or unsupervised classifiers with a mean average error (MAE) of residuals lower than 0.1 for many classification tasks. The predictors are publicly available in a Python library whose usage is straightforward and does not require domain-specific skill or expertise. 

## Usage

Examples can be found in the 'debug' folder. In a nutshell, all you need is to prepare your dataset as features and labels, initialize a FRAPPEObject and call the 'predict_metric function'. Here you find an example of code usage from one of the available code snippets.

https://github.com/tommyippoz/FRAPPE/blob/7c9f7781021b875cd0c511347ef612104d3fe9d7/debug/regression_test.py#L41-L50
