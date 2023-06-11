import numpy as np


class BayesianLinearRegression:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha  # prior precision
        self.beta = beta    # observation noise precision
        self.w_mean = None  # posterior mean of weights
        self.w_cov = None   # posterior covariance of weights

    def fit(self, X, y):
        N, D = X.shape
        X_aug = np.concatenate((np.ones((N, 1)), X), axis=1)
        prior_cov = np.linalg.inv(self.alpha * np.eye(D+1))
        self.w_cov = np.linalg.inv(prior_cov + self.beta * X_aug.T @ X_aug)
        self.w_mean = self.beta * self.w_cov @ X_aug.T @ y

    def predict(self, X):
        N, D = X.shape
        X_aug = np.concatenate((np.ones((N, 1)), X), axis=1)
        y_mean = X_aug @ self.w_mean
        y_var = 1 / self.beta + np.sum(X_aug @ self.w_cov * X_aug, axis=1)
        return y_mean, y_var

    def sample_posterior(self, X, num_samples):
        N, D = X.shape
        X_aug = np.concatenate((np.ones((N, 1)), X), axis=1)
        weights_samples = np.random.multivariate_normal(self.w_mean, self.w_cov, num_samples)
        y_samples = X_aug @ weights_samples.T

        return y_samples
