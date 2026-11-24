import os
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bike Sharing Demand Predictor",
    page_icon="🚲",
    layout="centered",
)

# -----------------------------------------------------------------------------
# 2. LOAD THE TRAINED MODEL BUNDLE
# -----------------------------------------------------------------------------
MODEL_PATH = "bike_model_bundle.pkl"


@st.cache_resource
def load_model_bundle(path: str):
    """Loads the saved scikit-learn model and feature metadata once."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


bundle = load_model_bundle(MODEL_PATH)
# -----------------------------------------------------------------------------
# 3. HEADER SECTION
# -----------------------------------------------------------------------------
st.title("🚲 Bike Sharing Demand Predictor")
st.markdown(
    "Estimate daily bicycle rental demand using a **Multiple Linear Regression** "
    "model trained on the UCI Bike Sharing dataset."
)

if bundle is None:
    st.error(
        f"Model file `{MODEL_PATH}` not found! Please run **Cell 10** of your "
        "`bike_sharing_regression.py` notebook first to generate the model file, "
        "then place it in the same folder as `app.py`."
    )
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
metrics = bundle.get("metrics", {})

# Display model accuracy metrics in the sidebar
with st.sidebar:
    st.header("📊 Model Performance")
    st.caption("Evaluated on 20% Test Split (Multiple Linear Regression)")
    if metrics:
        st.metric("R² Score", f"{metrics.get('R2_Score', 0) * 100:.1f}%")
        st.metric("MAE (Avg Error)", f"{metrics.get('MAE', 0):.0f} bikes")
        st.metric("RMSE", f"{metrics.get('RMSE', 0):.0f} bikes")
    st.divider()
    st.markdown(
        "**Project:** Bike Sharing Demand Prediction  \n"
        "**Dataset:** UCI Bike Sharing (`day.csv`)"
    )

# -----------------------------------------------------------------------------
# 4. USER INPUT FORM
# -----------------------------------------------------------------------------
st.subheader("Enter Day & Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    # Real-world weather inputs (normalized automatically for the model)
    temp_c = st.slider("Temperature (°C)", min_value=0.0, max_value=41.0, value=22.0, step=0.5)
    hum_pct = st.slider("Humidity (%)", min_value=0, max_value=100, value=60, step=1)
    wind_kmh = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=67.0, value=12.0, step=0.5)

    weather_map = {
        "Clear / Partly Cloudy": 1,
        "Mist / Cloudy": 2,
        "Light Snow / Rain": 3,
    }
    weather_label = st.selectbox("Weather Condition", list(weather_map.keys()))
    weathersit = weather_map[weather_label]

with col2:
    season_map = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
    season_label = st.selectbox("Season", list(season_map.keys()), index=1)
    season = season_map[season_label]
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_label = st.selectbox("Month", month_names, index=5)
    mnth = month_names.index(month_label) + 1

    weekday_names = [
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
    ]
    weekday_label = st.selectbox("Day of Week", weekday_names, index=3)
    weekday = weekday_names.index(weekday_label)

    yr_map = {"2011 (Year 0)": 0, "2012 (Year 1)": 1}
    yr_label = st.selectbox("Reference Year", list(yr_map.keys()), index=1)
    yr = yr_map[yr_label]

col3, col4 = st.columns(2)
with col3:
    holiday = 1 if st.checkbox("Public Holiday?", value=False) else 0
with col4:
    default_working = 1 if (weekday not in [0, 6] and holiday == 0) else 0
    workingday = 1 if st.checkbox("Working Day?", value=bool(default_working)) else 0

# -----------------------------------------------------------------------------
# 5. PREPROCESS INPUTS & PREDICT
# -----------------------------------------------------------------------------
def prepare_input_dataframe() -> pd.DataFrame:
    """
    Normalizes continuous weather inputs using UCI dataset scaling factors
    and constructs the exact one-hot encoded feature vector expected by the model.
    """
    # Start with all expected columns initialized to 0
    row = {col: 0 for col in feature_columns}

    # Normalize continuous features to match day.csv scaling
    row["temp"] = temp_c / 41.0
    row["hum"] = hum_pct / 100.0
    row["windspeed"] = wind_kmh / 67.0
    row["yr"] = yr
    row["holiday"] = holiday
    row["workingday"] = workingday
    # Activate corresponding one-hot encoded categorical columns (drop_first=True)
    for prefix, val in [
        ("season", season),
        ("mnth", mnth),
        ("weekday", weekday),
        ("weathersit", weathersit),
    ]:
        col_name = f"{prefix}_{val}"
        if col_name in row:
            row[col_name] = 1

    return pd.DataFrame([row], columns=feature_columns)
st.divider()

if st.button("Predict Bike Rental Demand", type="primary", use_container_width=True):
    try:
        input_df = prepare_input_dataframe()
        raw_prediction = model.predict(input_df)[0]
        # Clamp negative edge-case predictions to 0 rentals
        predicted_bikes = max(0, int(round(raw_prediction)))

        st.success(f"### Estimated Daily Demand: **{predicted_bikes:,} bicycles**")

        # Provide a quick contextual demand level indicator
        if predicted_bikes < 2500:
            st.info("📉 **Demand Level: Low** — Lighter fleet availability needed.")
        elif predicted_bikes < 5500:
            st.info("⚖️ **Demand Level: Moderate** — Standard daily bike distribution.")
        else:
            st.info("📈 **Demand Level: High** — Peak demand! Ensure maximum fleet availability.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")