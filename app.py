# Education Loan Eligibility Prediction Using Machine Learning
# Save as: app.py
#
# Install:
# pip install streamlit pandas numpy scikit-learn matplotlib
#
# Run:
# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Education Loan Eligibility",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Education Loan Eligibility Prediction")
st.write(
    "Predict whether an education loan application is "
    "eligible using Machine Learning."
)

st.caption(
    "Educational project demonstration. Predictions should not "
    "be used as the sole basis for real lending decisions."
)

# ---------------------------------------------------------
# FILE SETTINGS
# ---------------------------------------------------------

DATA_FILE = Path("education_loan_data.csv")

FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

TARGET = "Loan_Status"

# ---------------------------------------------------------
# SAMPLE DATA CREATION
# ---------------------------------------------------------

def create_dataset(rows=700, seed=42):

    rng = np.random.default_rng(seed)

    gender = rng.choice(
        ["Male", "Female"],
        rows,
        p=[0.70, 0.30]
    )

    married = rng.choice(
        ["Yes", "No"],
        rows,
        p=[0.60, 0.40]
    )

    dependents = rng.choice(
        ["0", "1", "2", "3+"],
        rows,
        p=[0.55, 0.18, 0.17, 0.10]
    )

    education = rng.choice(
        ["Graduate", "Not Graduate"],
        rows,
        p=[0.75, 0.25]
    )

    self_employed = rng.choice(
        ["Yes", "No"],
        rows,
        p=[0.15, 0.85]
    )

    applicant_income = rng.integers(
        1500,
        15000,
        rows
    )

    coapplicant_income = np.round(
        rng.uniform(0, 8000, rows),
        0
    )

    loan_amount = np.round(
        rng.uniform(20, 500, rows),
        0
    )

    loan_term = rng.choice(
        [120, 180, 240, 300, 360, 480],
        rows,
        p=[0.03, 0.04, 0.08, 0.08, 0.72, 0.05]
    )

    credit_history = rng.choice(
        [1.0, 0.0],
        rows,
        p=[0.82, 0.18]
    )

    property_area = rng.choice(
        ["Urban", "Semiurban", "Rural"],
        rows
    )

    # Create an approximate eligibility score
    score = (
        credit_history * 4
        + (education == "Graduate") * 1
        + (married == "Yes") * 0.3
        + (applicant_income > 5000) * 1
        + (coapplicant_income > 2000) * 0.5
        + (loan_amount < 250) * 0.8
        + (property_area == "Semiurban") * 0.3
        + rng.normal(0, 1, rows)
    )

    loan_status = np.where(
        score >= 4.0,
        "Y",
        "N"
    )

    data = pd.DataFrame({
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status
    })

    data.to_csv(DATA_FILE, index=False)

    return data


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

def load_data():

    if not DATA_FILE.exists():
        return create_dataset()

    data = pd.read_csv(DATA_FILE)

    missing_columns = [
        column
        for column in FEATURES + [TARGET]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    return data


# ---------------------------------------------------------
# BUILD MACHINE LEARNING MODEL
# ---------------------------------------------------------

def create_model():

    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area"
    ]

    numerical_features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        max_depth=10,
        min_samples_leaf=2
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

@st.cache_resource
def train_model(data_signature):

    data = load_data()

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return (
        model,
        X_test,
        y_test,
        predictions,
        accuracy,
        matrix,
        report
    )


# ---------------------------------------------------------
# LOAD AND TRAIN
# ---------------------------------------------------------

try:

    data = load_data()

    signature = (
        str(DATA_FILE.resolve())
        + str(DATA_FILE.stat().st_mtime_ns)
        + str(data.shape)
    )

    (
        model,
        X_test,
        y_test,
        predictions,
        accuracy,
        matrix,
        report
    ) = train_model(signature)

except Exception as error:

    st.error(
        f"Error loading or training model: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Loan Prediction",
        "Dataset",
        "Graphs",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.write(
    f"Dataset Records: {len(data)}"
)

st.sidebar.write(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# PAGE 1 - LOAN PREDICTION
# ---------------------------------------------------------

if page == "Loan Prediction":

    st.header("🎓 Education Loan Prediction")

    st.write(
        "Enter the applicant information below."
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000,
            step=500
        )

    with col2:

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0,
            value=2000,
            step=500
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1,
            value=150,
            step=10
        )

        loan_term = st.selectbox(
            "Loan Amount Term",
            [120, 180, 240, 300, 360, 480],
            index=4
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x:
                "Good (1)" if x == 1.0
                else "Poor (0)"
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    st.divider()

    if st.button(
        "🔍 Check Loan Eligibility",
        type="primary",
        use_container_width=True
    ):

        applicant = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        result = model.predict(
            applicant
        )[0]

        probability = model.predict_proba(
            applicant
        )[0]

        classes = model.named_steps[
            "classifier"
        ].classes_

        probability_dict = dict(
            zip(classes, probability)
        )

        if result == "Y":

            st.success(
                "✅ Loan Eligibility: ELIGIBLE"
            )

            st.metric(
                "Eligibility Probability",
                f"{probability_dict.get('Y', 0) * 100:.2f}%"
            )

        else:

            st.error(
                "❌ Loan Eligibility: NOT ELIGIBLE"
            )

            st.metric(
                "Eligibility Probability",
                f"{probability_dict.get('N', 0) * 100:.2f}%"
            )

        st.subheader("Application Details")

        st.dataframe(
            applicant,
            use_container_width=True
        )

        result_data = applicant.copy()

        result_data["Prediction"] = result

        result_data["Probability"] = (
            max(probability) * 100
        )

        st.download_button(
            "Download Prediction",
            data=result_data.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="loan_prediction.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------
# PAGE 2 - DATASET
# ---------------------------------------------------------

elif page == "Dataset":

    st.header("📊 Education Loan Dataset")

    st.write(
        "Dataset used for training and testing the model."
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        len(data)
    )

    col2.metric(
        "Eligible",
        int(
            (data[TARGET] == "Y").sum()
        )
    )

    col3.metric(
        "Not Eligible",
        int(
            (data[TARGET] == "N").sum()
        )
    )

    st.subheader("Missing Values")

    missing_values = data.isnull().sum()

    st.dataframe(
        missing_values.rename(
            "Missing Values"
        ),
        use_container_width=True
    )

    st.download_button(
        "⬇ Download Dataset",
        data=data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="education_loan_data.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# PAGE 3 - GRAPHS
# ---------------------------------------------------------

elif page == "Graphs":

    st.header("📈 Data Visualization")

    # Loan status graph
    st.subheader(
        "Loan Eligibility Distribution"
    )

    status_count = (
        data[TARGET]
        .value_counts()
        .rename(
            index={
                "Y": "Eligible",
                "N": "Not Eligible"
            }
        )
    )

    fig1, ax1 = plt.subplots()

    ax1.bar(
        status_count.index,
        status_count.values
    )

    ax1.set_xlabel(
        "Loan Status"
    )

    ax1.set_ylabel(
        "Number of Applicants"
    )

    ax1.set_title(
        "Education Loan Eligibility"
    )

    st.pyplot(fig1)

    plt.close(fig1)

    # Education graph
    st.subheader(
        "Education Distribution"
    )

    education_count = (
        data["Education"]
        .value_counts()
    )

    fig2, ax2 = plt.subplots()

    ax2.bar(
        education_count.index,
        education_count.values
    )

    ax2.set_xlabel(
        "Education"
    )

    ax2.set_ylabel(
        "Applicants"
    )

    ax2.set_title(
        "Applicants by Education"
    )

    st.pyplot(fig2)

    plt.close(fig2)

    # Income vs Loan Amount
    st.subheader(
        "Applicant Income vs Loan Amount"
    )

    fig3, ax3 = plt.subplots()

    eligible = data[
        data[TARGET] == "Y"
    ]

    not_eligible = data[
        data[TARGET] == "N"
    ]

    ax3.scatter(
        eligible["ApplicantIncome"],
        eligible["LoanAmount"],
        label="Eligible",
        alpha=0.6
    )

    ax3.scatter(
        not_eligible["ApplicantIncome"],
        not_eligible["LoanAmount"],
        label="Not Eligible",
        alpha=0.6
    )

    ax3.set_xlabel(
        "Applicant Income"
    )

    ax3.set_ylabel(
        "Loan Amount"
    )

    ax3.legend()

    ax3.set_title(
        "Income and Loan Amount"
    )

    st.pyplot(fig3)

    plt.close(fig3)

    # Credit history
    st.subheader(
        "Credit History Distribution"
    )

    credit_count = (
        data["Credit_History"]
        .value_counts()
    )

    fig4, ax4 = plt.subplots()

    ax4.bar(
        ["Good Credit", "Poor Credit"],
        [
            credit_count.get(1.0, 0),
            credit_count.get(0.0, 0)
        ]
    )

    ax4.set_ylabel(
        "Applicants"
    )

    ax4.set_title(
        "Credit History"
    )

    st.pyplot(fig4)

    plt.close(fig4)


# ---------------------------------------------------------
# PAGE 4 - MODEL EVALUATION
# ---------------------------------------------------------

elif page == "Model Evaluation":

    st.header(
        "🤖 Machine Learning Model Evaluation"
    )

    st.write(
        "Algorithm: Random Forest Classifier"
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Confusion Matrix"
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Actual N",
                "Actual Y"
            ],
            columns=[
                "Predicted N",
                "Predicted Y"
            ]
        )

        st.dataframe(
            matrix_df,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Classification Report"
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

    st.subheader(
        "Prediction Results"
    )

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )


# ---------------------------------------------------------
# PAGE 5 - ABOUT
# ---------------------------------------------------------

elif page == "About Project":

    st.header(
        "📚 About the Project"
    )

    st.markdown("""
    ### Project Title

    **Education Loan Eligibility Prediction Using Machine Learning**

    ### Objective

    The objective of this project is to develop a machine
    learning system that predicts whether an applicant is
    eligible for an education loan based on application
    information.

    ### Machine Learning Algorithm

    **Random Forest Classifier**

    Random Forest combines multiple decision trees and uses
    their combined predictions for classification.

    ### Input Features

    - Gender
    - Married
    - Dependents
    - Education
    - Self Employed
    - Applicant Income
    - Coapplicant Income
    - Loan Amount
    - Loan Amount Term
    - Credit History
    - Property Area

    ### Output

    The system produces:

    - Eligible
    - Not Eligible
    - Eligibility probability

    ### Technologies

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit

    ### Project Workflow

    1. Collect education loan data.
    2. Load the CSV dataset.
    3. Clean and preprocess the data.
    4. Encode categorical variables.
    5. Split the dataset.
    6. Train the Random Forest model.
    7. Evaluate the model.
    8. Enter applicant details.
    9. Predict loan eligibility.
    10. Display the result and graphs.
    """)

    st.subheader(
        "CSV File Format"
    )

    st.code("""
Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,Loan_Status
Male,Yes,0,Graduate,No,5000,2000,150,360,1,Urban,Y
Female,No,0,Graduate,No,3500,0,100,360,1,Semiurban,Y
Male,No,2,Not Graduate,Yes,2500,1000,180,360,0,Rural,N
""", language="csv")

    st.info(
        "The application automatically creates a sample CSV "
        "when education_loan_data.csv is not present."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Education Loan Eligibility Prediction Using Machine Learning"
)

st.caption(
    "Run locally using: python -m streamlit run app.py"
)
