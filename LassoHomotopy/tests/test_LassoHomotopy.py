import pytest
import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopy


def test_fit():
    """Test fitting the model with random data."""
    np.random.seed(0)  # Set seed for reproducibility
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model = LassoHomotopy(tol=1e-4, max_iter=5000)
    model.fit(X, y, alpha=0.5)

    assert model.coef_ is not None
    assert len(model.coef_) == X.shape[1]


def test_prediction():
    """Test prediction with a fitted model."""
    np.random.seed(0)  # Set seed for reproducibility
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model = LassoHomotopy(tol=1e-4, max_iter=1000)
    model.fit(X, y, alpha=0.5)

    predictions = model.predict(X)
    assert predictions.shape == y.shape


def test_alpha_tuning():
    """Test automatic alpha tuning."""
    np.random.seed(0)  # Set seed for reproducibility
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model = LassoHomotopy(tol=1e-4, max_iter=1000)
    alphas = np.logspace(-4, 1, 10)  # Search for optimal alpha
    best_alpha = model.tune_alpha(X, y, alphas)
    print(f"Best alpha: {best_alpha}")

    assert best_alpha in alphas


def test_polynomial_features():
    """Test polynomial feature expansion."""
    np.random.seed(0)  # Set seed for reproducibility
    X = np.random.randn(100, 3)

    model = LassoHomotopy(tol=1e-4, max_iter=1000)
    X_poly = model.add_polynomial_features(X, degree=3)

    assert X_poly.shape[1] > X.shape[1]


def test_visualization():
    """Test if the coefficient plot executes without errors."""
    np.random.seed(0)  # Set seed for reproducibility
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model = LassoHomotopy(tol=1e-4, max_iter=1000)
    model.fit(X, y, alpha=0.5)

    # Ensure the visualization does not raise an exception
    try:
        model.plot_coefficients()
    except Exception as e:
        pytest.fail(f"Visualization failed with error: {e}")
