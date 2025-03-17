import numpy as np


class LassoHomotopyModel:
    def __init__(self, tol=1e-6, max_iter=1000):
        """
        Initialize the LASSO model using the Homotopy Method.
        :param tol: Tolerance for stopping criteria.
        :param max_iter: Maximum number of iterations.
        """
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None  # Model coefficients
        self.lambda_vals = []  # Stores lambda values along the path
        self.active_set = []  # Indices of active features

    def soft_thresholding(self, x, lambda_):
        """Apply soft-thresholding to enforce sparsity in LASSO."""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)  # Initialize coefficients to zero
        residual = y.copy()  # Initial residual
        lambda_val = np.max(np.abs(X.T @ y))  # Start with max lambda
        self.lambda_vals = [lambda_val]  # Ensure it's a list (not empty)

        for _ in range(self.max_iter):
            correlations = X.T @ residual
            j_max = np.argmax(np.abs(correlations))

            if j_max not in self.active_set:
                self.active_set.append(j_max)

            X_active = X[:, self.active_set]
            coef_active = np.linalg.lstsq(X_active, y, rcond=None)[0]

            # Apply soft thresholding to enforce sparsity
            coef_active = self.soft_thresholding(coef_active, lambda_val)

            self.coef_ = np.zeros(n_features)
            self.coef_[self.active_set] = coef_active.flatten()

            residual = y - X @ self.coef_

            # Ensure lambda update
            lambda_new = (
                np.min(np.abs(correlations[self.active_set])) if self.active_set else 0
            )

            # Check for convergence
            if len(self.lambda_vals) > 1 and np.abs(lambda_val - lambda_new) < self.tol:
                break  # Stop if change is below tolerance

            # Update lambda value
            lambda_val = lambda_new
            self.lambda_vals.append(lambda_val)  # Append new lambda

        return LassoHomotopyResults(self.coef_)


class LassoHomotopyResults:
    def __init__(self, coef_):
        """
        Store results of the LASSO model.
        :param coef_: Learned coefficients
        """
        self.coef_ = coef_

    def predict(self, X):
        """
        Predict target values using the fitted model.
        :param X: Feature matrix (n_samples, n_features)
        :return: Predicted values (n_samples,)
        """
        return X @ self.coef_
