import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import joblib
import os

MODEL_PATH = "models/"

model_names = {
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
    # "LightGBM": "lgbm_model.pkl",
    # "Ensemble": "ensemble_model.pkl"
}

selected_features = ['sbytes', 'dbytes', 'sttl', 'sload', 'dload', 'ackdat', 'smean', 'ct_state_ttl', 'ct_srv_dst']

# Load pre-trained models
label_encoder = joblib.load("utils/label_encoder.pkl")
# preprocessor = joblib.load("utils/preprocessor.pkl")

@st.cache_resource
def load_model(model_name):
    return joblib.load(os.path.join(MODEL_PATH, model_names[model_name]))

# === Sidebar Configuration ===
st.sidebar.header("📁 Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload UNSW-NB15 Dataset", type=["csv"])

# === Load Data ===
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ File loaded successfully!")
else:
    st.sidebar.warning("⚠️ Please upload UNSW-NB15 dataset.")
    st.stop()

# === Page Config ===
st.set_page_config(page_title="UNSW-NB15 Dashboard", layout="wide")
st.title("🛡️ Intrusion Detection Dashboard - UNSW-NB15")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Model Performance", "🧠 Attack Classification"])

# === Tab 1: Overview ===
with tab1:
    st.header("📊 Dataset Overview (Exploratory)")

    st.subheader("📌 Dataset Info")
    st.write("Shape:", df.shape)
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Label Distribution")

    if 'label' in df.columns:
        df['label'] = df['label'].map({0: 'Attack', 1: 'Normal'})
        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Count']

        fig_label = px.pie(label_counts, names='Label', values='Count',
                           title='Traffic Label Distribution')
        st.plotly_chart(fig_label, use_container_width=True)
    else:
        st.error("❌ Column 'label' not found in the dataset. Please upload a proper UNSW-NB15 CSV file.")
        
    if 'attack_cat' in df.columns:
        st.subheader("🧨 Attack Category Distribution")

        cat_counts = df['attack_cat'].value_counts().reset_index()
        cat_counts.columns = ['Attack Category', 'Count']

        fig_label = px.pie(cat_counts, names='Attack Category', values='Count',
                           title='Attack Category Distribution')
        st.plotly_chart(fig_label, use_container_width=True)
        
    # === Feature Distribution Explorer ===
    st.subheader("🔬 Feature Distribution Explorer")
    feature = st.selectbox("Select a feature to visualize:", selected_features)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        chart_type = st.radio("Choose chart type", ["Histogram", "Box Plot"])

    with col2:
        bins = st.slider("Number of bins (for histogram)", min_value=10, max_value=100, value=30)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=feature, color='label', nbins=bins, title=f"{feature} Distribution by Label")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        fig = px.box(df, x='label', y=feature, color='label', points="outliers",
                     title=f"{feature} Box Plot by Label")
        st.plotly_chart(fig, use_container_width=True)

# === Tab 2: Model Performance ===
with tab2:
    st.header("🎯 Model Performance")

    st.subheader("📌 Choose a Model")
    selected_model_name = st.selectbox("Select model to evaluate:", list(model_names.keys()))
    model = load_model(selected_model_name)
    
    st.subheader("📊 Predictions and Metrics")

    # Select only features used in modeling
    X = df[selected_features]
    
    # Define column types (optional, for debugging or dynamic pipelines)
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # True labels and prediction
    y_true = df['attack_cat']
    y_true_encoded = label_encoder.transform(y_true)  # numerical
    y_pred = model.predict(X)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Compute metrics
    accuracy = accuracy_score(y_true_encoded, y_pred)
    sensitivity = recall_score(y_true_encoded, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true_encoded, y_pred)
    specificity_list = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    mean_specificity = np.mean(specificity_list)

    try:
        y_proba = model.predict_proba(X)
        y_true_binarized = label_binarize(y_true_encoded, classes=np.arange(len(label_encoder.classes_)))
        roc_auc = roc_auc_score(y_true_binarized, y_proba, average='macro', multi_class='ovr')
    except:
        roc_auc = np.nan

    # === Show Summary Metrics ===
    st.subheader("📊 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", f"{accuracy:.4f}")
    col2.metric("📈 Sensitivity (Recall)", f"{sensitivity:.4f}")
    col3.metric("📉 Specificity (Avg)", f"{mean_specificity:.4f}")
    col4.metric("🧠 ROC AUC (OvR)", f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A")

    # === Misclassification Analysis ===
    st.subheader("🧠 Misclassified Samples")

    df['y_true'] = y_true # decoded strings
    df['y_pred'] = y_pred_decoded

    # List of all class labels (sorted for UI)
    classes = sorted(df['y_true'].unique())

    # False Negatives: attacks predicted as Normal
    fn_df = df[(df['y_true'] != 'Normal') & (df['y_pred'] == 'Normal')]
    st.markdown(f"🚨 **False Negatives (Attacks predicted as Normal)**: {len(fn_df)} cases")
    st.dataframe(fn_df.head(20), use_container_width=True)

    # Inspect specific class misclassifications
    st.subheader("🔍 Inspect Misclassified Class")
    selected_class = st.selectbox("Choose true label to inspect:", classes, key="tab2_selectbox")
    misclassified_class = df[(df['y_true'] == selected_class) & (df['y_pred'] != selected_class)]

    st.markdown(f"⚠️ **{selected_class} misclassified as other classes:** {len(misclassified_class)} samples")
    st.dataframe(misclassified_class.head(20), use_container_width=True)

    # Summary confusion stats per class
    st.markdown("### 🔍 Confusion Stats Summary")
    cm_summary = pd.DataFrame({
        "Class": classes,
        "Total True": [sum(df['y_true'] == cls) for cls in classes],
        "Correctly Predicted": [sum((df['y_true'] == cls) & (df['y_pred'] == cls)) for cls in classes],
        "Misclassified": [sum((df['y_true'] == cls) & (df['y_pred'] != cls)) for cls in classes],
    })
    st.dataframe(cm_summary, use_container_width=True)

    st.markdown("### 🧩 Confusion Matrix")

    fig_cm = px.imshow(
        cm,  # confusion matrix from earlier
        x=classes,
        y=classes,
        labels=dict(x="Predicted", y="True", color="Count"),
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix for {selected_model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# === Tab 3: Misclassifications ===
with tab3:
    st.header("🧠 Manual Attack Classification")

    st.subheader("📌 Select Model for Prediction")
    selected_model_name_tab3 = st.selectbox("Choose a model:", list(model_names.keys()), key="tab3_model_select")
    model = load_model(selected_model_name_tab3)

    st.subheader("📝 Input Feature Values")

    # Create input widgets for each feature
    user_input = {}
    for feature in selected_features:
        # Determine sensible default and range
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())

        user_input[feature] = st.slider(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100
        )

    # Convert input to dataframe
    input_df = pd.DataFrame([user_input])

    st.subheader("🔍 Input")
    st.dataframe(input_df, use_container_width=True)

    # Predict button
    if st.button("🔮 Predict Attack Type"):
        # Predict
        y_pred = model.predict(input_df)
        y_pred_label = label_encoder.inverse_transform(y_pred)[0]

        # Predict probabilities
        try:
            probs = model.predict_proba(input_df)[0]
            prob_df = pd.DataFrame({
                "Class": label_encoder.classes_,
                "Probability": probs
            }).sort_values("Probability", ascending=False)
        except:
            prob_df = None

        # Show result
        st.success(f"🛡️ **Predicted Class:** {y_pred_label}")

        if prob_df is not None:
            st.subheader("📊 Prediction Probabilities")
            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)
