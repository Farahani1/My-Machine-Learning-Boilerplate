import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    """
    A This function can be used to collect statistic from a multi-class or multi-label classification problem.
    """

    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=["y", "pred"])
        df.loc[:, "y"] = y[:, c]
        df.loc[:, "pred"] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(
                n=int(fold_size * (1 - prevalence)), replace=True
            )

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate(
                [pos_sample.pred.values, neg_sample.pred.values]
            )
            score = roc_auc_score(
                y_sample, pred_sample
            )  # You can use any other desired score
            statistics[c][i] = score
    return statistics


def confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, 0.95, axis=1)[i]
        min_ = np.quantile(statistics, 0.05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df
