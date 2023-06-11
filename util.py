import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from BLR import BayesianLinearRegression


def corrMat(df, id=False):
    corr_mat = df.corr().round(2)
    _, _ = plt.subplots(figsize=(10, 10))
    mask = np.zeros_like(corr_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr_mat, mask=mask, vmin=-1, vmax=1, center=0,
                cmap='plasma', square=False, lw=2, annot=True, cbar=False);
    plt.show()


def snsPairGrid(df):
    g = sns.PairGrid(df, diag_sharey=False)
    g.fig.set_size_inches(14, 13)
    g.map_diag(sns.kdeplot, lw=2)  # draw kde approximation on the diagonal
    g.map_lower(sns.scatterplot, s=15, edgecolor="k", linewidth=1, alpha=0.4)  # scattered plot on lower half
    g.map_lower(sns.kdeplot, cmap='plasma', n_levels=10)  # kde approximation on lower half
    plt.tight_layout()
    plt.show()


def modelEval(ldf, tdf, feature='median_house_value', scaling_id=False):
    # Given a dataframe, split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]  # remove target variable
    y_test = tdf[feature].copy()
    X_test = tdf.copy()
    del X_test[feature]

    model = BayesianLinearRegression()

    if scaling_id:
        X = (X - X.mean()) / X.std()  # standardize
        y = (y - y.mean()) / y.std()  # standardize
        X_test = (X_test - X_test.mean()) / X_test.std()
        y_test = (y_test - y_test.mean()) / y_test.std()
    # Fit Model
    model.fit(X, y)
    # Predict

    X_global = [X, X_test]
    y_global = [np.array(y), np.array(y_test)]
    names = ['Training', 'Test']

    for data, label, name in zip(X_global, y_global, names):
        y_pred = model.sample_posterior(data, 1)
        y_mean, y_std = model.predict(data)
        score = np.mean((label - y_pred) ** 2)
        print(name)
        print("Score:", score.round(2))
        print("Std:", y_std.round(2))
        print("Mean:", y_mean.round(2))

    return model

