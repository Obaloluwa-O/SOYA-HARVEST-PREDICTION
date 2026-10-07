"""
Step 1: fit the models and save everything the Streamlit app needs to one joblib file.

Run once (and again whenever the data changes):
    python train_model.py
"""
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
FILE_PATH = "Soya Again.xlsx"
TARGET = "Harvest2"
OUT_PATH = "soya_model.joblib"

FEATURE_SETS = {
    "X1": ["FS2", "Fert2", "OPV2"],
    "X2": ["FS2", "Fert2", "seeds2"],
    "X3": ["FS2", "Fert2", "OPV2", "pesticide2"],
    "X4": ["FS2", "Fert2", "seeds2", "pesticide2"],
}

# How the columns in the Excel file were logged before import.
# "log1p" = log(x + 1) (safe for zeros), "log" = natural log.
# Change this ONE line if you logged differently; the app reads it from the bundle.
TRANSFORM = "log1p"

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df = pd.read_excel(FILE_PATH).fillna(0)
Y = df[TARGET]

bundle = {
    "target": TARGET,
    "transform": TRANSFORM,
    "n_rows": int(len(df)),
    "feature_sets": FEATURE_SETS,
    "models": {},
}

for name, cols in FEATURE_SETS.items():
    X = df[cols]

    # 1) Honest out-of-sample check (same split as your notebook)
    X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_tr, y_tr)
    test_pred = lr.predict(X_te)

    # 2) Final model for inference: OLS on the full data (coefs, p-values, intervals)
    ols = sm.OLS(Y, sm.add_constant(X)).fit()

    order = np.argsort(y_te.values)
    bundle["models"][name] = {
        "features": cols,
        # what the app predicts with
        "params": ols.params,                    # incl. 'const'
        "cov": ols.cov_params(),                 # for prediction intervals
        "mse_resid": float(ols.mse_resid),
        "df_resid": int(ols.df_resid),
        # what the app shows
        "coef_table": pd.DataFrame(
            {
                "Coefficient": ols.params,
                "Std. error": ols.bse,
                "p-value": ols.pvalues,
                "CI low (95%)": ols.conf_int()[0],
                "CI high (95%)": ols.conf_int()[1],
            }
        ),
        "r2_full": float(ols.rsquared),
        "adj_r2_full": float(ols.rsquared_adj),
        "train_r2": float(lr.score(X_tr, y_tr)),
        "test_r2": float(r2_score(y_te, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, test_pred))),
        # chart data (sorted by actual, like your notebook plot)
        "test_actual": y_te.values[order],
        "test_pred": test_pred[order],
        # training ranges, to warn about extrapolation
        "feat_min": X.min(),
        "feat_max": X.max(),
        "feat_median": X.median(),
    }
    m = bundle["models"][name]
    print(f"{name}: train R2={m['train_r2']:.3f}  test R2={m['test_r2']:.3f}  test RMSE={m['test_rmse']:.3f}")

joblib.dump(bundle, OUT_PATH)
print(f"\nSaved -> {OUT_PATH}")
