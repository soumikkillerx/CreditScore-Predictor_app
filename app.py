import streamlit as st
import pickle
import pandas as pd

# Load models
with open('model_risk_rating.pkl', 'rb') as file:
    model_risk = pickle.load(file)
with open('model_credit_score.pkl', 'rb') as file:
    model_score = pickle.load(file)

st.title("Financial Risk Assessment App")
st.write("""
This app predicts your financial risk and credit score based on the information you provide.
Please fill out the details below to get the predictions.
""")

# User inputs
def user_input_features():
    st.sidebar.header("User Input Parameters")
    
    age = st.sidebar.slider("Age", 18, 100, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    education = st.sidebar.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    income = st.sidebar.slider("Income ($)", 0, 200000, 50000, step=5000)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
    loan_amount = st.sidebar.slider("Loan Amount ($)", 0, 100000, 10000, step=1000)
    loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Personal", "Business", "Education", "Home", "Auto"])
    employment_status = st.sidebar.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Retired"])
    years_at_job = st.sidebar.slider("Years at Current Job", 0, 50, 5)
    payment_history = st.sidebar.selectbox("Payment History", ["Good", "Bad"])
    dti_ratio = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01)
    assets_value = st.sidebar.slider("Assets Value ($)", 0, 1000000, 50000, step=10000)
    dependents = st.sidebar.slider("Number of Dependents", 0, 20, 0)
    previous_defaults = st.sidebar.slider("Previous Defaults", 0, 10, 0)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Marital Status': marital_status,
        'Income': income,
        'Credit Score': credit_score,
        'Loan Amount': loan_amount,
        'Loan Purpose': loan_purpose,
        'Employment Status': employment_status,
        'Years at Current Job': years_at_job,
        'Payment History': payment_history,
        'Debt-to-Income Ratio': dti_ratio,
        'Assets Value': assets_value,
        'Number of Dependents': dependents,
        'Previous Defaults': previous_defaults
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

# Display predictions
if st.button("Predict"):
    risk_rating = model_risk.predict(input_df)[0]
    credit_score = model_score.predict(input_df)[0]
    
    st.subheader("Predictions")
    st.write(f"Risk Rating: **{risk_rating}**")
    st.write(f"Credit Score: **{credit_score}**")
    
   

st.sidebar.subheader("About")
st.sidebar.info("This application uses a machine learning model to predict financial risk and credit score based on user inputs. Adjust the parameters to see how they affect the predictions.")
