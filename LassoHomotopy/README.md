# LASSO Regularized Regression (Homotopy Method)

## Description
This project implements LASSO regression using the **Homotopy Method**, an efficient algorithm for computing sparse solutions by iteratively adjusting the regularization parameter λ. The model is built from scratch using NumPy/SciPy and is ideal for high-dimensional datasets where feature selection and interpretability are critical.

## Key Features
- **Sparse Solutions:** Identifies and retains only the most influential features.
- **Collinear Data Handling:** Selects one feature from collinear groups and zeros others.
- **Efficient Regularization Path:** Computes solutions for a single λ value with active set updates.
- **Customizable Parameters:** Fine-tune regularization strength, convergence criteria, and preprocessing.

## Installation & Setup
### 1. Clone the Repository
`git clone <YOUR_FORKED_REPO_URL>`

`cd Project1`

### 2. Create a Virtual Environment
  `python3 -m venv venv`
  
  `source venv/bin/activate`        # On Windows: .\venv\Scripts\activate

### 3. Install Dependencies
  `pip install -r requirements.txt`

## Testing the Model
  `pytest`

### For Detailed Report
  `pytest -v`

  `pytest --cov=LassoHomotopy --cov-report=term-missing`

  `pytest -s --cov=LassoHomotopy --cov-report=term-missing`

## Parameters
| Parameter     | Type    | Default | Description                                                        |
| :------------ | :------ | :------ | :----------------------------------------------------------------- |
| `alpha`       | float   | `1.0`   | Regularization strength. Higher values increase sparsity.          |
| `max_iter`    | int     | `10000` | Maximum iterations for convergence.                                |
| `tolerance`   | float   | `1e-4`  | Stopping criterion (stops if λ < tolerance).                     |
| `standardize` | bool    | `True`  | Standardize features (mean=0, std=1) before fitting.            |
| `fit_intercept` | bool    | `True`  | Whether to fit an intercept term.                                |


## Test Coverage
* Collinear Data: `test_collinear_feature_selection` ensures only one feature is selected from collinear pairs.
* Sparsity: `test_sparsity_with_high_alpha` and `test_alpha_effect_on_sparsity` validate sparsity increases with higher *α*.
* Edge Cases: Zero inputs `test_edge_case_zero_input`, invalid dimensions `test_invalid_dimensions`.
* Predictions: Shape validation `test_predict_basic`.


## Limitations
* **Collinear Features**: The model selects one feature from collinear groups but may behave unpredictably with near-perfect collinearity.
* **High-Dimensional Data**: Performance may degrade with feature counts >10k due to matrix operations.
* **Zero Variance Features**: Features with no variability can cause numerical instability (preprocess to remove).


# Homotopy LASSO Regression

## 1. What does the model you have implemented do and when should it be used?
The **Homotopy LASSO** efficiently computes sparse solutions by iteratively updating the regularization parameter λ. It’s ideal for:
* Feature selection in high-dimensional data.
* Datasets with collinear features.
* Scenarios requiring interpretable models (e.g., identifying key predictors).

## 2. How did you test your model to determine if it is working reasonably correctly?
* **Collinear Data**: Verified sparsity and single-feature selection from collinear groups.
* **Edge Cases**: Zero inputs, invalid dimensions, and large *α* values.
* **Parameter Sensitivity**: Validated α’s impact on sparsity.

## 3. What parameters have you exposed to users of your implementation in order to tune performance? 
Adjust:
`alpha`: Increase for sparser solutions.
`max_iter`: Raise for large datasets.
`tolerance`: Lower for higher precision (at computational cost).

## 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
**Current Challenges:**
* **Singular Matrices**: Addressed with ridge regularization but may fail with extreme collinearity.
* **Computational Cost**: Scales with feature count (avoid for >10k features without optimization).

**Potential Future Improvements:**
* **Support for Warm Starts:** Implementing the ability to provide a "warm start" (i.e., initializing the algorithm with the solution from a previous run with a similar `alpha` value) could drastically accelerate the computation of the regularization path, especially when fitting the model for a sequence of `alpha` values.
* **Enhanced Numerical Stability for Edge Cases:** Implementing checks and robust handling mechanisms for edge cases like near-zero denominators or zero-variance features would improve the model's reliability and prevent potential errors.
