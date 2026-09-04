import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="IoT Intrusion Detection System", layout="wide")

st.sidebar.image(
    "assets/samsung_logo.svg",
    width=150
)

st.markdown("""
<style>

/* =========================================================
   SAMSUNG COLOR PALETTE
   ========================================================= */

:root {
    --samsung-blue: #1428A0;
    --samsung-input: #1738B8;
    --samsung-dark: #0B1F66;
    --white: #FFFFFF;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background-color: #1428A0 !important;
}

/* Normal sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}


/* =========================================================
   SAMSUNG LOGO
   ========================================================= */

section[data-testid="stSidebar"] img {
    filter: brightness(0) invert(1) !important;
}


/* =========================================================
   MODEL SELECTION BOX
   WHITE BOX + BLUE TEXT
   ========================================================= */

section[data-testid="stSidebar"]
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 10px !important;
}

/* Selected model text */
section[data-testid="stSidebar"]
div[data-baseweb="select"] span {
    color: #1428A0 !important;
}

/* Selected model text — additional BaseWeb elements */
section[data-testid="stSidebar"]
div[data-baseweb="select"] div {
    color: #1428A0 !important;
}

/* Arrow */
section[data-testid="stSidebar"]
div[data-baseweb="select"] svg {
    color: #1428A0 !important;
    fill: #1428A0 !important;
}


/* =========================================================
   MODEL DROPDOWN
   WHITE BACKGROUND + BLUE TEXT
   ========================================================= */

div[role="listbox"] {
    background-color: #FFFFFF !important;
}

div[role="option"] {
    background-color: #FFFFFF !important;
    color: #1428A0 !important;
}

div[role="option"] * {
    color: #1428A0 !important;
}

div[role="option"]:hover {
    background-color: #E8ECFF !important;
}


/* =========================================================
   RADIO BUTTONS
   ========================================================= */

section[data-testid="stSidebar"]
div[role="radio"] {
    color: #FFFFFF !important;
    background-color: transparent !important;
    border-radius: 10px !important;
    padding: 8px 10px !important;
    margin: 3px 0 !important;
}

section[data-testid="stSidebar"]
div[role="radio"] * {
    color: #FFFFFF !important;
}

/* Selected radio */
section[data-testid="stSidebar"]
div[role="radio"][aria-checked="true"] {
    background-color: #0B1F66 !important;
}


/* =========================================================
   NUMBER INPUTS
   BLUE BOX + WHITE TEXT
   ========================================================= */

div[data-testid="stNumberInput"]
div[data-baseweb="input"] {
    background-color: #1738B8 !important;
    border: 2px solid #1428A0 !important;
    border-radius: 10px !important;
}

div[data-testid="stNumberInput"] input {
    background-color: #1738B8 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}


/* =========================================================
   + / - BUTTONS
   ========================================================= */

div[data-testid="stNumberInput"] button {
    background-color: #0B1F66 !important;
    color: #FFFFFF !important;
    border: none !important;
}

div[data-testid="stNumberInput"] button svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}


/* =========================================================
   INPUT LABELS
   ========================================================= */

div[data-testid="stNumberInput"] label {
    color: #1428A0 !important;
    font-weight: 600 !important;
}


/* =========================================================
   PAGE TITLE
   ========================================================= */

h1 {
    color: #1428A0 !important;
}


/* =========================================================
   TABS
   ========================================================= */

button[data-baseweb="tab"] {
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1428A0 !important;
}
/* =========================================================
   PERFORMANCE METRICS TABLE
   ========================================================= */

.metrics-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 15px;
}

.metrics-table th {
    background-color: #0B1F66 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    text-align: center !important;
    padding: 12px !important;
}

.metrics-table td {
    background-color: #1738B8 !important;
    color: #FFFFFF !important;
    text-align: center !important;
    padding: 12px !important;
    border-top: 1px solid #1428A0 !important;
}

.metrics-table tbody tr:hover td {
    background-color: #1428A0 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("IoT Intrusion Detection System")
st.caption("AI-powered network traffic analysis and intrusion detection")

# -----------------------------------------------------------------------------
# 1. Preset Values Definition
# -----------------------------------------------------------------------------
NORMAL_PRESET = {
    "Src Port": 33344, "Dst Port": 443, "Protocol": 6, "Flow Duration": 379933,
    "Total Fwd Packet": 11, "Total Length of Fwd Packet": 720.0,
    "Fwd Packet Length Min": 0.0, "Flow IAT Min": 1.0, "Bwd Packets/s": 28.95,
    "Packet Length Std": 669.52, "RST Flag Count": 0, "FWD Init Win Bytes": 29200,
    "Bwd Init Win Bytes": 131, "FIN Flag Count": 1, "SYN Flag Count": 2
}


ATTACK_PRESET = {
    "Src Port": 36498.0,
    "Dst Port": 5910.0,
    "Protocol": 6.0,
    "Flow Duration": 2798.0,
    "Total Fwd Packet": 1.0,
    "Total Length of Fwd Packet": 0.0,
    "Fwd Packet Length Min": 0.0,
    "Flow IAT Min": 2798.0,
    "Bwd Packets/s": 357.39814152966403,
    "Packet Length Std": 0.0,
    "RST Flag Count": 1.0,
    "FWD Init Win Bytes": 1024.0,
    "Bwd Init Win Bytes": 0.0,
    "FIN Flag Count": 0.0,
    "SYN Flag Count": 1.0,
}

for key, val in NORMAL_PRESET.items():
    if f"feat_{key}" not in st.session_state:
        st.session_state[f"feat_{key}"] = val

def update_preset_values():
    selection = st.session_state.preset_selection
    if selection == "Sample Normal (Row 0)":
        target = NORMAL_PRESET
    elif selection == "Sample Attack":
        target = ATTACK_PRESET
    else:
        return
    for k, v in target.items():
        st.session_state[f"feat_{k}"] = v

# -----------------------------------------------------------------------------
# 2. Model & Scaler Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join("models", model_filename)
    model = joblib.load(model_path)
    if isinstance(model, LogisticRegression) and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model

scaler = load_scaler()

# Sidebar: Model Selection
st.sidebar.header("Model Configuration")
model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Ridge Classifier": "ridge_classifier.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib"
}

selected_model_name = st.sidebar.selectbox("Choose Classification Model", list(model_options.keys()))
model = load_model(model_options[selected_model_name])

st.sidebar.subheader("Testing Presets")
st.sidebar.radio(
    "Load Sample Data",
    ("Manual Input", "Sample Normal (Row 0)", "Sample Attack"),
    key="preset_selection",
    on_change=update_preset_values
)

# -----------------------------------------------------------------------------
# 3. Streamlit Tabs Setup (Prediction vs. Evaluation Dashboard)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Live Prediction", "Model Evaluation & Metrics"])

with tab1:
    st.subheader("Network Flow Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        src_port = st.number_input("Src Port", min_value=0, max_value=65535, key="feat_Src Port")
        dst_port = st.number_input("Dst Port", min_value=0, max_value=65535, key="feat_Dst Port")
        protocol = st.number_input("Protocol", min_value=0, max_value=255, key="feat_Protocol")
        flow_duration = st.number_input("Flow Duration", min_value=0, key="feat_Flow Duration")
        tot_fwd_pkt = st.number_input("Total Fwd Packet", min_value=0, key="feat_Total Fwd Packet")

    with col2:
        tot_len_fwd_pkt = st.number_input("Total Length of Fwd Packet", key="feat_Total Length of Fwd Packet")
        fwd_pkt_len_min = st.number_input("Fwd Packet Length Min", key="feat_Fwd Packet Length Min")
        flow_iat_min = st.number_input("Flow IAT Min", key="feat_Flow IAT Min")
        bwd_pkts_s = st.number_input("Bwd Packets/s", key="feat_Bwd Packets/s")
        pkt_len_std = st.number_input("Packet Length Std", key="feat_Packet Length Std")

    with col3:
        rst_flag_cnt = st.number_input("RST Flag Count", min_value=0, key="feat_RST Flag Count")
        fwd_init_win_bytes = st.number_input("FWD Init Win Bytes", min_value=0, key="feat_FWD Init Win Bytes")
        bwd_init_win_bytes = st.number_input("Bwd Init Win Bytes", min_value=0, key="feat_Bwd Init Win Bytes")
        fin_flag_cnt = st.number_input("FIN Flag Count", min_value=0, key="feat_FIN Flag Count")
        syn_flag_cnt = st.number_input("SYN Flag Count", min_value=0, key="feat_SYN Flag Count")

    input_data = pd.DataFrame([{
        "Src Port": src_port, "Dst Port": dst_port, "Protocol": protocol,
        "Flow Duration": flow_duration, "Total Fwd Packet": tot_fwd_pkt,
        "Total Length of Fwd Packet": tot_len_fwd_pkt, "Fwd Packet Length Min": fwd_pkt_len_min,
        "Flow IAT Min": flow_iat_min, "Bwd Packets/s": bwd_pkts_s, "Packet Length Std": pkt_len_std,
        "RST Flag Count": rst_flag_cnt, "FWD Init Win Bytes": fwd_init_win_bytes,
        "Bwd Init Win Bytes": bwd_init_win_bytes, "FIN Flag Count": fin_flag_cnt,
        "SYN Flag Count": syn_flag_cnt
    }])

    st.markdown("---")

    if st.button("Predict Network Status", type="primary", use_container_width=True):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            prob_normal, prob_attack = probabilities[0] * 100, probabilities[1] * 100
        else:
            prob_normal, prob_attack = None, None

        if prediction == 1:
            st.error(f"**Threat Detected: ATTACK (Class 1)** using *{selected_model_name}*")
        else:
            st.success(f"**Normal Traffic (Class 0)** using *{selected_model_name}*")

        if prob_normal is not None:
            c1, c2 = st.columns(2)
            c1.metric("Normal Confidence", f"{prob_normal:.2f}%")
            c2.metric("Attack Confidence", f"{prob_attack:.2f}%")

with tab2:
    st.subheader("Performance Metrics Across All Models")
    st.markdown("Comparison of Train vs. Test metrics for Accuracy, Precision, Recall, and F1-Score.")

    # TODO: Replace these placeholder numbers with your actual metrics from your notebook training phase!
    metrics_data = [
        {"Model": "Random Forest",       "Dataset": "Train", "Accuracy": 0.99, "Precision": 0.99, "Recall": 0.99, "F1-Score": 0.99},
        {"Model": "Random Forest",       "Dataset": "Test",  "Accuracy": 0.97, "Precision": 0.96, "Recall": 0.97, "F1-Score": 0.96},
        {"Model": "Decision Tree",       "Dataset": "Train", "Accuracy": 0.99, "Precision": 0.99, "Recall": 0.99, "F1-Score": 0.99},
        {"Model": "Decision Tree",       "Dataset": "Test",  "Accuracy": 0.95, "Precision": 0.94, "Recall": 0.95, "F1-Score": 0.94},
        {"Model": "Logistic Regression", "Dataset": "Train", "Accuracy": 0.92, "Precision": 0.90, "Recall": 0.91, "F1-Score": 0.90},
        {"Model": "Logistic Regression", "Dataset": "Test",  "Accuracy": 0.91, "Precision": 0.89, "Recall": 0.90, "F1-Score": 0.89},
        {"Model": "Naive Bayes",         "Dataset": "Train", "Accuracy": 0.85, "Precision": 0.82, "Recall": 0.86, "F1-Score": 0.84},
        {"Model": "Naive Bayes",         "Dataset": "Test",  "Accuracy": 0.84, "Precision": 0.81, "Recall": 0.85, "F1-Score": 0.83},
        {"Model": "Ridge Classifier",    "Dataset": "Train", "Accuracy": 0.89, "Precision": 0.88, "Recall": 0.87, "F1-Score": 0.87},
        {"Model": "Ridge Classifier",    "Dataset": "Test",  "Accuracy": 0.88, "Precision": 0.87, "Recall": 0.86, "F1-Score": 0.86},
    ]

    metrics_df = pd.DataFrame(metrics_data)

    # Convert the DataFrame to an HTML table
    table_html = metrics_df.to_html(
        index=False,
        classes="metrics-table",
        border=0
    )

    st.markdown(table_html, unsafe_allow_html=True)
