import os
import numpy as np
import csv
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel


def test_predict():
    model = LassoHomotopyModel()
    data = []

    # Dynamically select the dataset
    file_name = "collinear_data.csv"  # Change this to "collinear_data.csv" or "small_test.csv" to test different datasets
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

    assert os.path.exists(file_path), f"Test data file not found: {file_path}"

    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(f"Row: {row}")  # Debugging line
            data.append(row)

    # Check if data is loaded
    assert len(data) > 0, "Dataset is empty!"

    # Detect feature columns based on dataset type
    if file_name == "collinear_data.csv":
        feature_prefix = "X_"
        target_column = "target"
    elif file_name == "diabetes.csv":
        feature_prefix = ""  # No prefix in diabetes.csv
        target_column = "Outcome"
    elif file_name == "small_test.csv":
        feature_prefix = "x"  # No prefix in diabetes.csv
        target_column = "y"
    else:
        raise ValueError("Unsupported dataset!")

    # Extract features and target
    X = np.array(
        [
            [
                float(v)
                for k, v in datum.items()
                if k.startswith(feature_prefix) and k != target_column
            ]
            for datum in data
        ]
    )
    y = np.array([float(datum[target_column]) for datum in data])

    print(f"X shape: {X.shape}, y shape: {y.shape}")  # Debugging prints
    print(f"X: {X}")

    assert X.size > 0 and y.size > 0, "X or y is empty!"

    results = model.fit(X, y)


def test_lambda_not_decreasing():
    """Test that the algorithm stops when lambda_new is not decreasing."""
    X = np.array([[1, 2], [2, 4], [3, 6]])  # Perfectly collinear
    y = np.array([1, 2, 3])

    model = LassoHomotopyModel()
    model.fit(X, y)

    # The algorithm should stop early due to lambda_new >= lambda_val
    assert len(model.lambda_vals) < model.max_iter, "Lambda values did not stop early"


def test_lambda_converges():
    """Test that the algorithm stops when lambda change is below tolerance."""
    X = np.array([[1, 0], [0, 1]])  # Independent features
    y = np.array([0.001, 0.001])  # Small target values

    model = LassoHomotopyModel(tol=1e-4)
    model.fit(X, y)

    # Debug: Print lambda values
    print("Lambda Values:", model.lambda_vals)

    # Compute changes in lambda values
    lambda_changes = [
        abs(model.lambda_vals[i] - model.lambda_vals[i - 1])
        for i in range(1, len(model.lambda_vals))
    ]

    # Debug: Print lambda changes
    print("Lambda Changes:", lambda_changes)

    # The lambda sequence should stop when change is small
    assert any(
        change < model.tol for change in lambda_changes
    ), "Lambda change did not fall below tolerance"


def test_non_collinear_case():
    """Test Lasso on independent features"""
    X = np.array([[1, 0], [0, 1]])  # Fully independent features
    y = np.array([1, -1])

    model = LassoHomotopyModel()
    result = model.fit(X, y)

    assert result.coef_ is not None, "Model did not return coefficients!"
