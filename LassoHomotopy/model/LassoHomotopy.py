import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin

class LassoHomotopy(BaseEstimator, RegressorMixin):
    def __init__(self, tol=1e-4, max_iter=1000, warm_start=False, use_cache=True):
        self.tol = tol
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.use_cache = use_cache
        self.coef_ = None
        self.active_set_ = set()
        self.cached_active_set_ = set() if use_cache else None
        self.history = []

    def fit(self, X, y, alpha):
        n_samples, n_features = X.shape
        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros(n_features)

        residual = y - X @ self.coef_
        active_set = set(self.active_set_) if self.use_cache else set()

        for iteration in range(self.max_iter):
            grad = -X.T @ residual
            max_grad_idx = np.argmax(np.abs(grad))

            if np.abs(grad[max_grad_idx]) < alpha + self.tol:
                break

            active_set.add(max_grad_idx)
            self.active_set_ = active_set

            # Solve the smaller problem with the active set only
            X_active = X[:, list(active_set)]
            coef_active = np.linalg.lstsq(X_active, y, rcond=None)[0]

            # Update coefficients
            self.coef_ = np.zeros(n_features)
            self.coef_[list(active_set)] = coef_active

            residual = y - X @ self.coef_

            # Store history for visualization
            self.history.append(self.coef_.copy())

            # Check convergence
            if np.linalg.norm(residual) < self.tol:
                break

        # Cache the active set
        if self.use_cache:
            self.cached_active_set_ = self.active_set_

    def tune_alpha(self, X, y, alphas, cv=5):
        """Automatically tune the regularization parameter using cross-validation."""
        best_alpha = None
        best_score = float("inf")
        kf = KFold(n_splits=cv)

        for alpha in alphas:
            scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.fit(X_train, y_train, alpha)
                predictions = self.predict(X_test)
                score = np.mean((predictions - y_test) ** 2)
                scores.append(score)

            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        return best_alpha

    def add_polynomial_features(self, X, degree=2):
        """Add polynomial features to handle non-linear relationships."""
        poly = PolynomialFeatures(degree)
        return poly.fit_transform(X)

    def coordinate_descent_fallback(self, X, y, alpha):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        for iteration in range(self.max_iter):
            for j in range(n_features):
                X_j = X[:, j]
                rho = X_j @ (y - (X @ self.coef_) + self.coef_[j] * X_j)

                if rho < -alpha:
                    self.coef_[j] = (rho + alpha) / np.sum(X_j**2)
                elif rho > alpha:
                    self.coef_[j] = (rho - alpha) / np.sum(X_j**2)
                else:
                    self.coef_[j] = 0

    def predict(self, X):
        return X @ self.coef_

    def get_params(self):
        return {
            "tol": self.tol,
            "max_iter": self.max_iter,
            "warm_start": self.warm_start,
            "use_cache": self.use_cache,
        }

    def plot_coefficients(self):
        plt.figure(figsize=(12, 6))
        for i in range(len(self.history[0])):
            plt.plot(
                range(len(self.history)),
                [coef[i] for coef in self.history],
                label=f"Coef {i}",
            )
        plt.xlabel("Iteration")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficient Path During Homotopy Iterations")
        plt.legend()
        plt.show()

    def grid_search(self, X, y, param_grid, cv=5):
        grid_search = GridSearchCV(self, param_grid, cv=cv)
        grid_search.fit(X, y)
        return grid_search.best_params_, grid_search.best_score_

    def cross_val_score(self, X, y, cv=5):
        scores = []
        kf = KFold(n_splits=cv)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.fit(X_train, y_train)
            predictions = self.predict(X_test)
            score = np.mean((predictions - y_test) ** 2)
            scores.append(score)
        return np.mean(scores)
