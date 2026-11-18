
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef2ff, #ecfeff);
    color: #172554;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

h1 {
    color: #312e81 !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #3730a3 !important;
    font-weight: 700 !important;
}

p, label {
    color: #334155;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #312e81, #164e63);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.2);
}

/* Metric cards */

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #e0e7ff);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #c7d2fe;
    box-shadow: 0 5px 15px rgba(49,46,129,0.08);
}

div[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #4338ca !important;
    font-weight: 800;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #4f46e5, #0891b2);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 700;
    width: 100%;
    transition: 0.3s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #3730a3, #0e7490);
    transform: translateY(-2px);
}

/* Forms */

div[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #c7d2fe;
    box-shadow: 0 8px 25px rgba(49,46,129,0.08);
}

/* Input fields */

div[data-baseweb="input"] {
    border-radius: 10px;
    background: white;
}

/* Cards */

.info-card {
    background: white;
    padding: 23px;
    border-radius: 17px;
    border: 1px solid #c7d2fe;
    box-shadow: 0 6px 18px rgba(49,46,129,0.08);
    margin-bottom: 15px;
}

.hero {
    background: linear-gradient(120deg, #312e81, #4f46e5, #0891b2);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 30px rgba(49,46,129,0.2);
}

.hero h2 {
    color: white !important;
    font-size: 34px;
}

.hero p {
    color: #e0f2fe !important;
    font-size: 17px;
}

.main-subtitle {
    color: #64748b;
    font-size: 16px;
}

hr {
    border-color: #c7d2fe;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA AND TRAIN MODEL
# =====================================================

@st.cache_resource
def load_model():

    df = pd.read_csv("diabetes.csv")

    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    # Separate input and output
    x = df.drop(columns=["Outcome"]).copy()
    y = df["Outcome"]

    # Outlier handling using IQR clipping
    for col in df.select_dtypes(include=["int64", "float64"]).columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        if col in x.columns:
            x[col] = x[col].clip(lower_bound, upper_bound)

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=50
    )

    # Feature scaling
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Logistic Regression
    model = LogisticRegression()

    model.fit(x_train_scaled, y_train)

    # Predictions
    predictions = model.predict(x_test_scaled)

    probabilities = model.predict_proba(x_test_scaled)[:, 1]

    # Evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }

    return (
        df,
        features,
        model,
        scaler,
        x_test_scaled,
        y_test,
        predictions,
        probabilities,
        metrics,
        x
    )


try:
    (
        df,
        features,
        model,
        scaler,
        x_test_scaled,
        y_test,
        predictions,
        probabilities,
        metrics,
        x
    ) = load_model()

except Exception as e:
    st.error(f"Error loading dataset or training model: {e}")
    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
<div style="text-align:center;padding:15px 0;">
    <div style="font-size:55px;">🩺</div>
    <h2 style="color:white !important;">MediPredict AI</h2>
    <p style="color:#c7d2fe !important;">
        Smart Healthcare Analytics
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Diabetes Prediction",
        "Dataset Overview",
        "Model Performance",
        "Feature Importance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="background:rgba(255,255,255,0.1);
padding:15px;border-radius:12px;">

<h4 style="color:white;">🤖 Model Information</h4>

<p style="color:#e0e7ff;">
Algorithm: Logistic Regression<br>
Scaling: StandardScaler<br>
Task: Binary Classification<br>
Dataset: Diabetes Dataset
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.caption("Developed using Python and Streamlit")


# =====================================================
# HOME PAGE
# =====================================================


# =====================================================
# HOME PAGE
# =====================================================


if page == "Home":

    st.markdown("""
<div style="background-color:#f8fafc; padding:25px; border-radius:12px; text-align:center;">
<h2>🩺 Diabetes Prediction System</h2>

<p>
Smart Healthcare Analytics powered by
Artificial Intelligence and Machine Learning.
</p>

<p>
Predict • Analyze • Understand
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("## Welcome to MediPredict AI")

    st.markdown("""
    <p class="main-subtitle">
        An intelligent diabetes prediction application
        using Logistic Regression.
    </p>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3 = st.columns(3)

    c1.metric("📊 Total Records", len(df))
    c2.metric("🧬 Input Features", len(features))
    c3.metric("🤖 Algorithm", "Logistic Regression")

    st.markdown("---")

    st.markdown("## ✨ Explore the Application")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>🩺 Diabetes Prediction</h3>
            <p>
                Enter patient health parameters and
                generate an ML-based prediction.
            </p>
        </div>

        <div class="info-card">
            <h3>📊 Dataset Analysis</h3>
            <p>
                Explore dataset records, statistical
                summaries and feature correlations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>📈 Model Evaluation</h3>
            <p>
                Understand accuracy, precision, recall,
                F1-score and ROC-AUC.
            </p>
        </div>

        <div class="info-card">
            <h3>🧠 Feature Analysis</h3>
            <p>
                Explore how different health features
                influence model predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### About the Project")

    st.write("""
    This project uses eight medical parameters to predict
    the possibility of diabetes.

    The data is processed using outlier handling and
    StandardScaler before being passed to a
    Logistic Regression classification model.
    """)

    st.info(
        "This application is developed for academic and "
        "educational purposes only. It is not a substitute "
        "for professional medical diagnosis."
    )
    if page == "Home":

       st.markdown("""
       <div style="background-color:#f8fafc; padding:25px; border-radius:12px;">
        <h2>🩺 Diabetes Prediction System</h2>

        <p>
            Smart Healthcare Analytics powered by
            Artificial Intelligence and Machine Learning.
        </p>

        <p>
            Predict • Analyze • Understand
        </p>
        </div>
        """, unsafe_allow_html=True)



# =====================================================
# DIABETES PREDICTION PAGE
# =====================================================

elif page == "Diabetes Prediction":

    st.title("🩺 Diabetes Prediction")

    st.markdown(
        '<p class="main-subtitle">'
        'Enter the medical information below to generate a prediction.'
        '</p>',
        unsafe_allow_html=True
    )

    st.write("")

    with st.form("prediction_form"):

        st.subheader("👤 Patient Information")

        col1, col2 = st.columns(2)

        with col1:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=1
            )

            glucose = st.number_input(
                "Glucose Level",
                min_value=0.0,
                max_value=300.0,
                value=120.0
            )

            bp = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            skin = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=100.0,
                value=20.0
            )

        with col2:

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=900.0,
                value=80.0
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        st.markdown("---")

        submit = st.form_submit_button(
            "🔍 Predict Diabetes"
        )

    if submit:

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                bp,
                skin,
                insulin,
                bmi,
                dpf,
                age
            ]],
            columns=features
        )

        # Apply the same training-data IQR limits
        for col in features:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            input_data[col] = input_data[col].clip(
                lower,
                upper
            )

        # Scale input
        scaled_input = scaler.transform(input_data)

        # Prediction
        result = model.predict(scaled_input)[0]

        probability = model.predict_proba(
            scaled_input
        )[0][1]

        st.markdown("---")

        st.subheader("📋 Prediction Result")

        col1, col2 = st.columns(2)

        if result == 1:

            with col1:
                st.error(
                    "⚠️ Model Prediction: Positive"
                )

            with col2:
                st.metric(
                    "Predicted Class",
                    "Diabetic"
                )

        else:

            with col1:
                st.success(
                    "✅ Model Prediction: Negative"
                )

            with col2:
                st.metric(
                    "Predicted Class",
                    "Non-Diabetic"
                )

        st.markdown("### Prediction Probability")

        st.metric(
            "Estimated Probability of Positive Class",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.caption(
            "This is the model's estimated probability, "
            "not a confirmed diagnosis. Consult a qualified "
            "healthcare professional for medical advice."
        )


# =====================================================
# DATASET OVERVIEW PAGE
# =====================================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write(
        "Explore the diabetes dataset and understand "
        "the distribution of its features."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Outcome Distribution")

        counts = df["Outcome"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            ["Non-Diabetic", "Diabetic"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=["#6366f1", "#06b6d4"],
            width=0.6
        )

        ax.set_ylabel("Number of Records")
        ax.set_title("Diabetes Outcome Distribution")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.subheader("Feature Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(
            data=df,
            x="Glucose",
            hue="Outcome",
            kde=True,
            palette=["#6366f1", "#06b6d4"],
            ax=ax
        )

        ax.set_title("Glucose Distribution")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.write(
        "Evaluation of the Logistic Regression model "
        "on the testing dataset."
    )

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    metric_columns = [
        col1, col2, col3, col4, col5
    ]

    for col, (name, value) in zip(
        metric_columns,
        metrics.items()
    ):

        col.metric(
            name,
            f"{value * 100:.2f}%"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Non-Diabetic",
                "Diabetic"
            ],
            yticklabels=[
                "Non-Diabetic",
                "Diabetic"
            ],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.subheader("📉 ROC Curve")

        fpr, tpr, thresholds = roc_curve(
            y_test,
            probabilities
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.plot(
            fpr,
            tpr,
            color="#4f46e5",
            linewidth=3,
            label=f"AUC = {metrics['ROC-AUC']:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#94a3b8"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")

        ax.legend()

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )


# =====================================================
# FEATURE IMPORTANCE PAGE
# =====================================================

elif page == "Feature Importance":

    st.title("🧠 Feature Analysis")

    st.write(
        "Explore the influence of different features "
        "in the Logistic Regression model."
    )

    coefficients = model.coef_[0]

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients
    })

    coef_df["Absolute Coefficient"] = (
        coef_df["Coefficient"].abs()
    )

    coef_df = coef_df.sort_values(
        by="Absolute Coefficient",
        ascending=False
    )

    st.markdown("---")

    st.subheader("📊 Logistic Regression Coefficients")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#06b6d4" if value > 0 else "#6366f1"
        for value in coef_df["Coefficient"]
    ]

    ax.barh(
        coef_df["Feature"],
        coef_df["Coefficient"],
        color=colors
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Features")
    ax.set_title("Feature Coefficient Analysis")

    ax.invert_yaxis()

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    st.subheader("📋 Coefficient Details")

    st.dataframe(
        coef_df,
        use_container_width=True
    )

    st.info("""
    Positive coefficients increase the model's predicted
    odds of diabetes, while negative coefficients decrease
    the predicted odds.

    Larger absolute coefficient values indicate a stronger
    association within this fitted model, not necessarily
    causal importance.
    """)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")


st.markdown("""
<div style="text-align:center; padding:20px; margin-top:30px;">
    <h4 style="color:#4338ca;">🩺 MediPredict AI</h4>
    <p style="color:#64748b;">
        Diabetes Prediction System | Powered by Machine Learning
    </p>
    <p style="color:#94a3b8;font-size:12px;">
        Academic Project • Logistic Regression • Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

