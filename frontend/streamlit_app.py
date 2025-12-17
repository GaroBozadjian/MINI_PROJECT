import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ML Dashboard Demo", layout="wide")

API_BASE = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

st.title("ML Dashboard Demo (Streamlit → FastAPI)")

# --- Health check ---
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Check API health"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

st.divider()

# --- Analysis section ---
st.header("1) Data Analysis (from API)")
if st.button("Load analysis"):
    try:
        r = requests.get(f"{API_BASE}/analyze", timeout=10)
        r.raise_for_status()
        data = r.json()

        st.write(f"Rows: **{data['rows']}**, Columns: **{data['columns']}**")

        # describe is nested dict: column -> stats -> value
        describe_df = pd.DataFrame(data["describe"])
        st.dataframe(describe_df, use_container_width=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")

st.divider()

# --- Prediction section ---
st.header("2) Prediction (from API)")

with st.form("predict_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sepal_length = st.number_input("sepal_length", value=5.1)
    with c2:
        sepal_width = st.number_input("sepal_width", value=3.5)
    with c3:
        petal_length = st.number_input("petal_length", value=1.4)
    with c4:
        petal_width = st.number_input("petal_width", value=0.2)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        pred = r.json()

        st.subheader("Result")
        st.write(f"Predicted class: **{pred['class_name']}** (id={pred['class_id']})")

        probs = pd.DataFrame([pred["probabilities"]])
        st.dataframe(probs, use_container_width=True)

    except requests.HTTPError:
        st.error(f"Prediction failed: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
