"""
Step 2: the Streamlit app.   Run with:  streamlit run app.py
Needs soya_model.joblib (created by train_model.py) in the same folder.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Soya harvest predictor", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------
# Labels: edit these to match your survey units
# ----------------------------------------------------------------------------
LABELS = {
    "FS2": "Farm size (ha)",
    "Fert2": "Fertilizer (kg)",
    "OPV2": "OPV seed (kg)",
    "seeds2": "Seeds (kg)",
    "pesticide2": "Pesticide (litres)",
}
HARVEST_UNIT = "kg"

GREEN, OCHRE, INK, SAGE = "#3F6B2F", "#C8962E", "#1F2A1C", "#EEF2E6"

# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Instrument Sans', sans-serif; color: {INK}; }}
.stApp {{ background: {SAGE}; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif !important; color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: #E2E9D6; }}
.block-container {{ padding-top: 2rem; max-width: 1150px; }}

.stFormSubmitButton button {{
    background: {GREEN}; color: #fff; border: 0; border-radius: 8px;
    padding: .65rem 1.6rem; font-weight: 600; font-size: 1.05rem;
}}
.stFormSubmitButton button:hover {{ background: #2f5222; color: #fff; }}
.stFormSubmitButton button:focus-visible {{ outline: 3px solid {OCHRE}; outline-offset: 2px; }}
[data-testid="stForm"] {{ background: #fff; border: 1px solid #D5DEC6; border-radius: 12px; padding: 1.4rem; }}

.result {{
    background: {GREEN}; color: #F4F7EC; border-radius: 14px; padding: 1.6rem 1.8rem; margin: 1.2rem 0 .4rem 0;
}}
.result .big {{ font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 700; line-height: 1.1; color: #fff; }}
.result .line {{ font-size: 1.1rem; margin-bottom: .5rem; }}
.result .sub {{ font-size: .95rem; opacity: .85; margin-top: .6rem; }}
.stale {{ color: #6b7560; font-style: italic; margin-top: 1rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model bundle
# ----------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("soya_model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("soya_model.joblib not found. Run `python train_model.py` first, then restart the app.")
    st.stop()

TRANSFORM = bundle["transform"]
fwd = np.log1p if TRANSFORM == "log1p" else np.log     # raw -> log scale
inv = np.expm1 if TRANSFORM == "log1p" else np.exp     # log scale -> raw


def predict(model, x_log):
    """Point prediction and 95% prediction interval on the log scale."""
    x = np.concatenate([[1.0], x_log])
    mean = float(model["params"].values @ x)
    se = np.sqrt(float(x @ model["cov"].values @ x) + model["mse_resid"])
    t = stats.t.ppf(0.975, model["df_resid"])
    return mean, mean - t * se, mean + t * se


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    set_name = st.selectbox(
        "Feature set",
        list(bundle["feature_sets"]),
        format_func=lambda k: f"{k}: " + ", ".join(bundle["feature_sets"][k]),
    )
    model = bundle["models"][set_name]
    cols = model["features"]

    already_logged = st.toggle(
        "My inputs are already logged",
        value=False,
        help="Off: type normal values (e.g. 2.5 ha) and the app logs them for you. "
        "On: type values exactly as they appear in the logged dataset.",
    )

    st.divider()
    st.subheader("How good is it?")
    c1, c2 = st.columns(2)
    c1.metric("Test R²", f"{model['test_r2']:.2f}")
    c2.metric("Test RMSE", f"{model['test_rmse']:.2f}")
    st.caption(f"Fitted on {bundle['n_rows']:,} farms. RMSE is on the log-harvest scale.")

# ----------------------------------------------------------------------------
# Header + input form (nothing runs until the button is pressed)
# ----------------------------------------------------------------------------
st.title("Soya harvest predictor")
st.write("Enter the farm's inputs, then press **Predict harvest**. Changing a value does nothing until you press it.")

with st.form("predict_form"):
    grid = st.columns(len(cols))
    values = {}
    for col_ui, col in zip(grid, cols):
        median_log = float(model["feat_median"][col])
        default = median_log if already_logged else float(inv(median_log))
        values[col] = col_ui.number_input(
            LABELS.get(col, col),
            min_value=0.0,
            value=round(default, 2),
            step=0.1,
            key=f"{set_name}_{col}_{already_logged}",
        )
    submitted = st.form_submit_button("Predict harvest")

if submitted:
    x_log = np.array([values[c] if already_logged else fwd(values[c]) for c in cols], dtype=float)
    mean, lo, hi = predict(model, x_log)
    out_of_range = [
        LABELS.get(c, c)
        for c, v in zip(cols, x_log)
        if v < model["feat_min"][c] or v > model["feat_max"][c]
    ]
    st.session_state["result"] = {
        "set": set_name, "mean": mean, "lo": lo, "hi": hi,
        "inputs": {LABELS.get(c, c): values[c] for c in cols},
        "out_of_range": out_of_range,
    }

res = st.session_state.get("result")

if res and res["set"] == set_name:
    st.markdown(
        f"""
<div class="result">
  <div class="line">The expected harvest for this farm is about</div>
  <div class="big">{inv(res['mean']):,.0f} {HARVEST_UNIT}</div>
  <div class="sub">There is a 95% chance the actual harvest falls between
  {max(inv(res['lo']), 0):,.0f} and {inv(res['hi']):,.0f} {HARVEST_UNIT}.
  (Model {res['set']}, log-harvest prediction {res['mean']:.2f})</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if res["out_of_range"]:
        st.warning(
            "These inputs are outside the range the model was trained on, so treat the result with caution: "
            + ", ".join(res["out_of_range"])
        )
elif res:
    st.markdown(
        '<div class="stale">You switched the feature set. Press Predict harvest to update the result.</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Chart: actual vs predicted on the held-out test set
# ----------------------------------------------------------------------------
st.subheader("Actual vs predicted (test set)")

fig, ax = plt.subplots(figsize=(12, 4.8))
fig.patch.set_facecolor("#FFFFFF")
x = np.arange(len(model["test_actual"]))
ax.plot(x, model["test_actual"], color="#B23A2E", linewidth=2, label="Actual (sorted)")
ax.scatter(x, model["test_pred"], s=14, alpha=0.55, color=GREEN, label="Predicted")
if res and res["set"] == set_name:
    ax.axhline(res["mean"], color=OCHRE, linestyle="--", linewidth=2, label="Your prediction")
ax.set_xlabel("Test observations, sorted by actual harvest")
ax.set_ylabel("log harvest")
ax.set_title(f"{set_name}: linear model", loc="left", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False)
st.pyplot(fig, width="stretch")
plt.close(fig)

# ----------------------------------------------------------------------------
# Coefficients
# ----------------------------------------------------------------------------
st.subheader("Coefficients")
coef = model["coef_table"].copy()
coef.index = [("Intercept" if i == "const" else LABELS.get(i, i)) for i in coef.index]

left, right = st.columns([3, 2])
with left:
    st.dataframe(
        coef.style.format(
            {"Coefficient": "{:.4f}", "Std. error": "{:.4f}", "p-value": "{:.4f}",
             "CI low (95%)": "{:.4f}", "CI high (95%)": "{:.4f}"}
        ),
        width="stretch",
    )
with right:
    st.caption(
        "Fitted by OLS on the full dataset. Because both inputs and harvest are on the log scale, "
        "a coefficient reads roughly as: a 1% increase in that input goes with a coefficient-sized "
        "% change in harvest, holding the others fixed. A p-value above 0.05 means the data cannot "
        "separate that input's effect from zero."
    )
    st.metric("R² (full data)", f"{model['r2_full']:.3f}")
    st.metric("Adjusted R²", f"{model['adj_r2_full']:.3f}")
