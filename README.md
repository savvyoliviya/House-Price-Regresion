# House Price Prediction (Ames, Iowa)

Advanced regression project for the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The goal is to predict the final sale price of residential homes in Ames, Iowa from 79 descriptive features.

The work is split into two notebooks:

| Notebook | Focus |
|---|---|
| `Project_2_P1_House_Price_Regression.ipynb` | Exploratory data analysis (EDA) |
| `Project_2_P2_House_Price_Regression.ipynb` | Feature engineering, feature selection, modeling, and submission |

## Dataset

| File | Description |
|---|---|
| `train.csv` | 1,460 houses, 81 columns including the target `SalePrice` |
| `test.csv` | 1,459 houses, 80 columns (no `SalePrice`) |
| `data_description.txt` | Definition of every feature and its categorical codes |
| `submission.csv` | Predictions for the 1,459 test houses (`Id`, `SalePrice`) |

Features cover lot and zoning, building type and style, quality and condition ratings, basement, garage, porch and pool details, year built/remodeled/sold, and sale type and condition.

## Part 1: Exploratory Data Analysis

- Target analysis: `SalePrice` is right-skewed and not normal, so it needs a log transform.
- Key features reviewed: `OverallQual`, `YearBuilt`, `TotalBsmtSF`, `GrLivArea`, `TotRmsAbvGrd`.
- Correlation heatmaps to find the strongest predictors and redundant pairs (e.g. `TotalBsmtSF`/`1stFlrSF`, `GarageCars`/`GarageArea`, `TotRmsAbvGrd`/`GrLivArea`).
- Missing-data audit: which features are missing, and whether "missing" really means "feature absent" (no basement, garage, fireplace, etc.).
- Outlier analysis: two houses with very large `GrLivArea` but low price.
- Regression assumptions: normality, homoscedasticity, linearity.
- Skewness review of numeric features, temporal analysis of the year columns, and breakdowns of discrete, continuous and categorical features against `SalePrice`.

## Part 2: Feature Engineering and Modeling

**Cleaning and transformation**
1. Remove the two `GrLivArea > 4000` outliers with low prices.
2. Log-transform the target with `np.log1p`.
3. Combine train and test so transformations are applied consistently.
4. Impute missing values:
   - `LotFrontage`: median by neighborhood.
   - Basement, garage, fireplace and veneer fields: `0` or `"None"`.
   - `MSZoning`, `Electrical`, `KitchenQual`, `Exterior1st/2nd`, `SaleType`: mode.
   - `Functional`: `"Typ"`.
5. Drop `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `Utilities`, and redundant basement and garage numerics.
6. Treat `MSSubClass`, `OverallCond`, `YrSold` and `MoSold` as categorical, and label-encode ordinal-like columns.
7. Add `TotalSF` (basement + 1st floor + 2nd floor).
8. Apply a Box-Cox transform (`boxcox1p`, λ = 0.15) to skewed numeric features (|skew| > 0.75).
9. One-hot encode with `get_dummies`, then scale with `MinMaxScaler`.

**Feature sets compared**

| Set | Description | Features |
|---|---|---|
| `X_cat` | Keeps categorical garage/basement columns | 206 |
| `X_free` | Drops categorical garage/basement columns | 191 |
| `X_cat_lasso` | `X_cat` after Lasso feature selection | 83 |
| `X_free_lasso` | `X_free` after Lasso feature selection | 77 |

Extra-Trees-based selection is also demonstrated.

**Models**
- XGBoost, Gradient Boosting (Huber loss), LightGBM, and Extra Trees, evaluated with 7-fold cross-validation (RMSE on log price) and feature-importance plots.
- A **stacking regressor**: XGBoost, Gradient Boosting and Extra Trees as base learners, with Linear Regression as the meta-learner.
- The final submission uses the stacked model trained on `X_cat_lasso`.

## Results

Stacking regressor with a linear meta-learner, RMSE on log-transformed price (from the notebook's recorded runs):

| Feature set | R² | RMSE |
|---|---|---|
| `X_cat_lasso` (83 features) | 0.985 | 0.0485 |
| `X_free_lasso` (77 features) | 0.981 | 0.0545 |
| `X_cat` (206 features) | 0.982 | 0.0536 |
| `X_free` (191 features) | 0.983 | 0.0524 |

`X_cat_lasso` scored best and was used for the final predictions. Note that these scores come from predicting the same data the model was trained on, so they are optimistic. Cross-validated or leaderboard scores will be higher (worse).

## Getting started

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm jupyter
```

### Run

1. Place `train.csv` and `test.csv` in the same folder as the notebooks.
2. Run `Project_2_P1_House_Price_Regression.ipynb` for the analysis.
3. Run `Project_2_P2_House_Price_Regression.ipynb` to rebuild the features, train the models, and write `submission.csv`.

## Project structure

```
.
├── Project_2_P1_House_Price_Regression.ipynb   # EDA
├── Project_2_P2_House_Price_Regression.ipynb   # Feature engineering + modeling
├── train.csv
├── test.csv
├── data_description.txt
├── submission.csv
└── README.md
```

## Known issues and tips

- **Submission is on the log scale:** the model is trained on `log1p(SalePrice)`, but the notebook writes the raw predictions to `submission.csv` (values like 11.73). Kaggle expects dollar prices, so convert before submitting: `np.expm1(y_preds)`. The `submission.csv` in this repo currently contains log values.
- **Feature selection leakage:** Lasso selection and scaling are fitted on the full training set before cross-validation, which can make CV scores slightly optimistic.
- **Final scores are in-sample:** see the note under Results. Add a held-out or cross-validated score for the final stack.
- **Linear SVR and MLP:** the recorded runs for these meta-learners performed very poorly (large negative R²) and were left commented out.
- **Library versions:** some plotting calls (`sns.distplot`, `size=` in `pairplot`, `drop(..., 1)`) are deprecated or removed in newer seaborn/pandas versions.
- **Install cell:** the first cell of Part 2 runs `pip install` for XGBoost and LightGBM; skip it if they are already installed.

## Acknowledgements

This project builds on several public Kaggle kernels, including:

- [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
- [A study on Regression applied to the Ames dataset](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset)
- [Stacked Regressions to predict House Prices](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
- [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models)
- [How I made top 0.3% on a Kaggle competition](https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)

Dataset: Ames Housing data compiled by Dean De Cock.
