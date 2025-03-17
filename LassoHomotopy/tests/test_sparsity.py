import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel


def test_lasso_sparsity():
    """Test that the LASSO model produces sparse solutions for correlated features."""
    # Step 1: Generate synthetic data with highly correlated features
    np.random.seed(42)
    n_samples, n_features = 100, 5

    # Generate correlated features
    X_base = np.random.randn(n_samples, 1)
    X = np.hstack(
        [X_base + 0.01 * np.random.randn(n_samples, 1) for _ in range(n_features)]
    )

    # Generate target variable with sparse true coefficients
    true_coef = np.array([3, 0, 0, 5, 0])  # Only two nonzero coefficients
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)  # Add noise

    # Step 2: Fit the Lasso Homotopy model
    lasso_model = LassoHomotopyModel(tol=1e-4)
    lasso_model.fit(X, y)

    # Step 3: Check sparsity - how many nonzero coefficients?
    num_nonzero = np.sum(np.abs(lasso_model.coef_) > 1e-4)

    # Step 4: Ensure sparsity
    assert (
        num_nonzero <= 2
    ), f"Lasso should select only a sparse subset of correlated features, but found {num_nonzero}!"
