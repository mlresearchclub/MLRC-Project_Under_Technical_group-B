"""
AgriSense - Irrigation Event Monitor (Streamlit version)
Run with:  streamlit run app.py
"""

from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 1. PAGE SETTINGS  (replaces <head>, <title>, and the layout CSS)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AgriSense - Irrigation Monitor",
    page_icon="🌱",
    layout="wide",
)

# ---------------------------------------------------------------
# 2. CONSTANTS  (replaces the <option> lists in the HTML)
# ---------------------------------------------------------------
SECTORS = [1, 2, 3, 4, 5]
EVENT_TYPES = ["irrigation", "fertigation"]
STOP_REASONS = [
    "fertigation timeout",
    "valve closure",
    "time restriction",
    "humidity threshold",
]

# Which saved model to use. Put the .joblib file in the SAME folder as app.py.
#   decision_tree_set_A.joblib = all features (includes the rule columns)
#   decision_tree_set_B.joblib = without the rule columns
MODEL_FILE = Path(__file__).parent / "decision_tree_set_A.joblib"

# If a name in this app differs from a column name in training_data.csv,
# map it here:  "name in app": "name in training data"
RENAME = {
    "water_volume": "water_volume_l",
    "temperature": "temperature_c",
    "rain_last_24_hours": "rain_prev_24h_mm",
}

# Stop reason (dropdown text) -> one-hot column name in training_data.csv
STOP_COLUMNS = {
    "fertigation timeout": "stop_fertigation_timeout",
    "valve closure": "stop_physical_valve_closure",
    "time restriction": "stop_time_restriction",
    "humidity threshold": "stop_humidity_threshold",
}

# Keys of every input widget (used by the Reset button)
INPUT_KEYS = [
    "sector", "event_type", "initial_humidity", "final_humidity",
    "duration", "stop_reason", "temperature", "rain",
]


# ---------------------------------------------------------------
# 3. MODEL  (connected to your Decision Tree bundle)
# ---------------------------------------------------------------
@st.cache_resource  # loads the model ONCE, not on every rerun
def load_model():
    """Load the saved bundle: {"pipeline": tree, "threshold": t, "features": [...]}."""
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def explain_rule(event_data: dict):
    """Return a plain-language reason using the labelling rules."""
    is_fert = event_data.get("is_fertigation") == 1
    duration = event_data.get("duration_hours")
    initial_hum = event_data.get("initial_humidity")
    final_hum = event_data.get("final_humidity")
    
    if initial_hum is not None and final_hum is not None:
        if initial_hum < 0 or initial_hum > 100 or final_hum < 0 or final_hum > 100:
            return "Humidity reading is outside 0 to 100, so the sensor value looks invalid."
    
    if is_fert and duration is not None and duration > 0.5:
        return f"Fertigation ran {duration:.2f} h, over the 0.5 h limit."
    
    if not is_fert and duration is not None and duration > 6.355:
        return f"Irrigation ran {duration:.2f} h, over the 6.36 h limit."
    
    if not is_fert and duration is not None and duration >= 1.0:
        if initial_hum is not None and final_hum is not None:
            change = final_hum - initial_hum
            if change < -10:
                return f"Humidity dropped by {-change:.1f} during a {duration:.2f} h irrigation."
    
    return "No labelling rule was triggered."


def predict_event(event_data: dict):
    """
    Returns None            -> model file not found
            {"missing": []} -> app is missing columns the model needs
            {"label", "confidence", "explanation"} -> normal result
    """
    bundle = load_model()
    if bundle is None:
        return None

    tree = bundle["pipeline"]
    threshold = bundle["threshold"]
    features = bundle["features"]          # exact columns + order from training

    # Rename app keys to training names, then check nothing is missing
    row = {RENAME.get(k, k): v for k, v in event_data.items()}
    missing = [f for f in features if f not in row]
    if missing:
        return {"missing": missing}

    X = pd.DataFrame([row])[features]      # one row, training column order

    prob_anomaly = float(tree.predict_proba(X)[0, 1])
    is_anomaly = prob_anomaly >= threshold  # threshold chosen during training
    confidence = prob_anomaly if is_anomaly else 1 - prob_anomaly

    rule_reason = explain_rule(event_data)
    explanation = f"**Reason (from labelling rules):** {rule_reason}"
    return {
        "label": "Anomaly" if is_anomaly else "Normal",
        "confidence": confidence,
        "explanation": explanation,
    }


# ---------------------------------------------------------------
# 4. RESET LOGIC  (replaces the form "reset" event listener)
# ---------------------------------------------------------------
def reset_form():
    for key in INPUT_KEYS:
        st.session_state[key] = None
    now = datetime.now()
    st.session_state["start_date"] = now.date()
    st.session_state["start_time"] = now.time().replace(second=0, microsecond=0)
    st.session_state["event_data"] = None


# First time the app opens, set everything to its empty state
if "event_data" not in st.session_state:
    reset_form()

# ---------------------------------------------------------------
# 5. HEADER
# ---------------------------------------------------------------
head_left, head_right = st.columns([1, 1])
head_left.markdown("## 🌱 AgriSense")
head_right.markdown(
    "<p style='text-align:right;color:#68776e;margin-top:18px'>"
    "Irrigation Event Monitor</p>",
    unsafe_allow_html=True,
)
st.divider()

st.title("Irrigation & Fertigation Analysis")
st.caption("Enter event details to prepare the event for analysis.")

# Two columns, like the CSS grid (1.4fr / 0.8fr)
left, right = st.columns([1.4, 0.8], gap="large")

# ---------------------------------------------------------------
# 6. LEFT CARD: INPUT FORM
# ---------------------------------------------------------------
with left:
    with st.container(border=True):
        st.subheader("Event Details")

        c1, c2 = st.columns(2)
        sector = c1.selectbox(
            "Sector", SECTORS, index=None, placeholder="Select Sector",
            format_func=lambda s: f"Sector {s}", key="sector",
        )
        event_type = c2.selectbox(
            "Event Type", EVENT_TYPES, index=None, placeholder="Select Type",
            format_func=str.capitalize, key="event_type",
        )

        d1, d2 = st.columns(2)
        start_date = d1.date_input("Start Date", key="start_date")
        start_time = d2.time_input("Start Time", key="start_time")

        c3, c4 = st.columns(2)
        initial_humidity = c3.number_input(
            "Initial Humidity (%)", min_value=0.0, max_value=100.0,
            value=None, step=0.1, placeholder="Enter initial humidity",
            key="initial_humidity",
        )
        final_humidity = c4.number_input(
            "Final Humidity (%)", min_value=0.0, max_value=100.0,
            value=None, step=0.1, placeholder="Enter final humidity",
            key="final_humidity",
        )

        c5, c6 = st.columns(2)
        duration = c5.number_input(
            "Duration (hours)", min_value=0.0,
            value=None, step=0.1, placeholder="Enter duration",
            key="duration",
        )
        stop_reason = c6.selectbox(
            "Stop Reason", STOP_REASONS, index=None,
            placeholder="Select Reason", format_func=str.capitalize,
            key="stop_reason",
        )

        c7, c8 = st.columns(2)
        temperature = c7.number_input(
            "Temperature (°C)", value=None, step=0.1,
            placeholder="Enter temperature", key="temperature",
        )
        rain = c8.number_input(
            "Rain in Last 24 Hours (mm)", min_value=0.0,
            value=None, step=0.1, placeholder="Enter rainfall", key="rain",
        )

        # -------------------------------------------------------
        # 7. CALCULATED VALUES  (replaces calculateValues() in JS)
        # Streamlit reruns the script on every change, so these
        # update live with no event listeners.
        # -------------------------------------------------------
        st.markdown("##### Calculated Values")

        humidity_change = None
        if initial_humidity is not None and final_humidity is not None:
            humidity_change = final_humidity - initial_humidity

        water_volume = None
        if duration is not None and sector is not None:
            factor = 0.4 if sector == 5 else 1.1  # Sector 5 differs
            water_volume = duration * 60 * factor

        start_dt = datetime.combine(start_date, start_time)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Humidity Change",
                  "—" if humidity_change is None else f"{humidity_change:.1f}")
        r2.metric("Water Volume Estimate",
                  "—" if water_volume is None else f"{water_volume:.1f}")
        r3.metric("Start Hour", start_dt.hour)
        r4.metric("Start Month", start_dt.strftime("%B"))

        # -------------------------------------------------------
        # 8. BUTTONS
        # -------------------------------------------------------
        b1, b2 = st.columns([3, 1])
        prepare = b1.button("Prepare Event", type="primary",
                            use_container_width=True)
        b2.button("Reset", on_click=reset_form, use_container_width=True)

        if prepare:
            # Replaces the "required" attribute / reportValidity()
            required = {
                "Sector": sector, "Event Type": event_type,
                "Initial Humidity": initial_humidity,
                "Final Humidity": final_humidity, "Duration": duration,
                "Stop Reason": stop_reason, "Temperature": temperature,
                "Rain": rain,
            }
            missing = [name for name, val in required.items() if val is None]

            if missing:
                st.session_state["event_data"] = None
                st.error("Please fill in: " + ", ".join(missing))
            else:
                event_data = {
                    "sector": sector,
                    "event_type": event_type,
                    "start_date_time": start_dt.isoformat(),
                    "initial_humidity": initial_humidity,
                    "final_humidity": final_humidity,
                    "humidity_change": humidity_change,
                    "duration_hours": duration,
                    "water_volume": water_volume,
                    "stop_reason": stop_reason,
                    "temperature": temperature,
                    "rain_last_24_hours": rain,
                    "start_hour": start_dt.hour,
                    "start_month": start_dt.month,
                }
                # One-hot columns for sector (sector_1 ... sector_5)
                for i in SECTORS:
                    event_data[f"sector_{i}"] = 1 if sector == i else 0
                # Fertigation flag (training column is_fertigation: 1 or 0)
                event_data["is_fertigation"] = (
                    1 if event_type == "fertigation" else 0
                )
                # One-hot columns for stop reason (names from training data)
                for reason, column in STOP_COLUMNS.items():
                    event_data[column] = 1 if stop_reason == reason else 0

                st.session_state["event_data"] = event_data

# ---------------------------------------------------------------
# 9. RIGHT CARD: ANALYSIS RESULT
# ---------------------------------------------------------------
with right:
    with st.container(border=True):
        st.subheader("Analysis Result")
        st.caption("Model prediction will appear here after integration.")

        event_data = st.session_state["event_data"]

        if event_data is None:
            st.info("**Ready for Analysis**\n\n"
                    "Complete the form and click Prepare Event.")
        else:
            st.success("**Event Details Prepared**")
            result = predict_event(event_data)

            if result is None:
                # Shown if the .joblib file is not next to app.py
                st.warning(f"Model file not found: {MODEL_FILE.name}")
                st.markdown("**Awaiting Prediction**")
                st.metric("Confidence", "—")
            elif "missing" in result:
                st.error(
                    "The model needs columns this app does not create yet: "
                    + ", ".join(result["missing"])
                )
            else:
                if result["label"] == "Normal":
                    st.success(f"✅ {result['label']}")
                else:
                    st.error(f"⚠️ {result['label']}")
                st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                st.markdown("###### Explanation")
                st.write(result["explanation"])

            # Handy while developing: see exactly what the model receives
            with st.expander("Prepared event data (debug)"):
                st.json(event_data)

st.divider()
st.caption("AgriSense | Smart Agriculture")