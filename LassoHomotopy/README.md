# LASSO Regularized Regression (Homotopy Method)

## Description
This project implements LASSO regularized regression using the Homotopy Method from first principles, without relying on built-in models from libraries like SciKit-Learn. The Homotopy Method is an efficient algorithm for solving the LASSO problem, which promotes sparsity in the solution by driving less important coefficients to zero.

## Key Features
* Sparse solutions for high-dimensional and collinear data.
* Efficient handling of feature selection and regularization.
* Built from scratch using NumPy and Scipy without pre-built models.
* Includes unit tests with pytest and unittest.

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


# Homotopy LASSO Regression

## 1. What does the model you have implemented do and when should it be used?
This model implements **LASSO (Least Absolute Shrinkage and Selection Operator) regression** using the **Homotopy Method**. This approach provides an efficient way to compute the entire regularization path of the LASSO estimator.
**When should it be used?**
* **Feature selection in high-dimensional data:** The LASSO's inherent ability to drive the coefficients of irrelevant features to zero makes it powerful for identifying a sparse set of important predictors when dealing with a large number of potential features.
* **Datasets with collinear features:** In the presence of multicollinearity, standard linear regression can have unstable and difficult-to-interpret coefficients. LASSO tends to select one feature from a group of highly correlated features and shrink the others towards zero, leading to a more stable and interpretable model.
* **Scenarios where interpretable, sparse models are preferred:** If understanding which features are most influential is crucial (e.g., in scientific research, medical diagnosis, or business decision-making), the sparsity induced by LASSO can provide valuable insights.

## 2. How did you test your model to determine if it is working reasonably correctly?
The model's robustness and correctness have been validated through a comprehensive suite of tests using the `pytest` framework. These tests include:
* **Collinear Data Validation:** The model's ability to induce sparsity in the presence of collinear features was specifically tested using synthetic datasets designed with high levels of multicollinearity (`collinear_data.csv` serves as an example). The tests verify that as the regularization strength (`alpha`) increases, the number of non-zero coefficients decreases as expected.
* **Edge Case Handling:** Tests were implemented to ensure the model handles various edge cases gracefully, including:
    * **Zero Inputs:** Evaluating the model's behavior when provided with datasets containing no features or no samples.
    * **Invalid Dimensions:** Checking for proper error handling when input data dimensions are inconsistent or unexpected.
    * **Standardization:** Verifying the correct application and impact of the `standardize` parameter on the model's performance and stability.
* **Parameter Sensitivity Analysis:** The impact of the key tuning parameter `alpha` on the resulting model sparsity was systematically examined. Tests confirm that increasing `alpha` leads to a sparser model with more coefficients driven to zero.
* **Dataset Variety:** Tests were conducted using both small, controlled datasets and larger synthetic datasets to assess the model's performance across different scales and complexities.

## 3. What parameters have you exposed to users of your implementation in order to tune performance? 
The following parameters can be adjusted to fine-tune the model's behavior:
* **`alpha`:** (float, default=1.0)
    * Controls the strength of the L1 regularization penalty.
    * A higher `alpha` leads to stronger regularization, resulting in a sparser model with more coefficients shrunk towards zero.
    * A lower `alpha` reduces the regularization effect, making the model closer to ordinary least squares.
* **`tolerance`:** (float, default=1e-4)
    * Specifies the tolerance for convergence in the Homotopy algorithm.
    * A smaller `tolerance` generally leads to more accurate results but may require more iterations.
* **`max_iter`:** (int, default=1000)
    * Sets the maximum number of iterations allowed for the Homotopy algorithm to converge.
    * Increasing `max_iter` may be necessary for large datasets or when a high level of precision is required.
* **`standardize`:** (bool, default=True)
    * Indicates whether to standardize the features (subtract the mean and divide by the standard deviation) before applying regularization.
    * Standardization is generally recommended to ensure that features with larger scales do not disproportionately influence the regularization process, improving numerical stability and model performance.

## 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
**Current Challenges:**
* **Computational Cost with Extremely High-Dimensional Data:** For datasets with a very large number of features (significantly exceeding the number of samples), the matrix operations involved in the Homotopy method can become computationally intensive and may lead to slower convergence times.
* **Instability with Zero Variance Features:** Features with zero variance (even after standardization) can lead to division by zero or other numerical instabilities during the algorithm's execution.

**Potential Future Improvements:**
* **Optimization of Matrix Inversions:** Exploring more efficient numerical linear algebra techniques, such as Cholesky decomposition where applicable, could significantly speed up the matrix inversion steps within the Homotopy algorithm.
* **Support for Warm Starts:** Implementing the ability to provide a "warm start" (i.e., initializing the algorithm with the solution from a previous run with a similar `alpha` value) could drastically accelerate the computation of the regularization path, especially when fitting the model for a sequence of `alpha` values.
* **Enhanced Numerical Stability for Edge Cases:** Implementing checks and robust handling mechanisms for edge cases like near-zero denominators or zero-variance features would improve the model's reliability and prevent potential errors.
