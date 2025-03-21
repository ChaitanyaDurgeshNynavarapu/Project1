import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopy


def test_sparse_solution():
    """Test that the model produces sparse coefficients with collinear data."""
    np.random.seed(42)

    n_samples, n_features = 100, 5

    # Create collinear data
    X_base = np.random.randn(n_samples, 1)
    X = np.hstack(
        [X_base + 0.01 * np.random.randn(n_samples, 1) for _ in range(n_features)]
    )

    true_coef = np.array([3.0, 0.0, 0.0, 5.0, 0.0])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    model = LassoHomotopy(tol=1e-4, max_iter=1000)
    model.fit(X, y, alpha=0.5)

    # Ensure some coefficients are zero
    assert np.count_nonzero(np.abs(model.coef_) > 1e-3) < n_features

    # Check for sparse coefficient vector
    assert np.count_nonzero(model.coef_) < n_features


def test_fallback():
    """Test the fallback to coordinate descent."""
    np.random.seed(42)

    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    model = LassoHomotopy(tol=1e-4, max_iter=1000)

    # Trigger fallback intentionally
    model.coordinate_descent_fallback(X, y, alpha=1e-10)

    assert np.all(np.isfinite(model.coef_)), "Fallback produced NaN coefficients"

    # Check for finite coefficients
    assert np.all(np.isfinite(model.coef_))
