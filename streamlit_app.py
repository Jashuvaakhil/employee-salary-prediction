import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Title
st.title("Employee Salary Prediction App")
st.write("This app predicts whether an individual's salary is >50K or <=50K based on demographic data.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")
    data['workclass'].replace({'?': 'Others'}, inplace=True)
    data['occupation'].replace({'?': 'Others'}, inplace=True)
    return data

data = load_data()

# Encode categorical features
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

df = preprocess(data)

# Train model
X = df.drop(['salary'], axis=1)
y = df['salary']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d0d0d;
        color: #f2f2f2;
        font-family: 'Segoe UI', sans-serif;
    }

    .main > div {
        background-color: #1a1a1a;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.05);
    }

    label, .stTextInput, .stSelectbox, .stNumberInput, .stSlider {
        color: #ffffff !important;
    }

    h1, h2, h3, h4 {
        color: #e0e0e0;
    }

.stButton > button {
    background-color: #87CEEB;  
    color: #000000;  
    font-size: 16px;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 8px;
    border: none;
    box-shadow: 0 4px 10px rgba(135, 206, 235, 0.5);
    transition: 0.3s ease;
}

.stButton > button:hover {
    background-color: #00BFFF;
    transform: scale(1.05);
    color: #ffffff;  
    box-shadow: 0 6px 14px rgba(0, 191, 255, 0.6);
}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 17, 90, 30)
workclass = st.sidebar.selectbox("Workclass", data['workclass'].unique())
fnlwgt = st.sidebar.number_input("Fnlwgt", value=200000)
education = st.sidebar.selectbox("Education", data['education'].unique())
education_num = st.sidebar.slider("Education Number", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", data['marital.status'].unique())
occupation = st.sidebar.selectbox("Occupation", data['occupation'].unique())
relationship = st.sidebar.selectbox("Relationship", data['relationship'].unique())
race = st.sidebar.selectbox("Race", data['race'].unique())
gender = st.sidebar.selectbox("Gender", data['gender'].unique())
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 99999, 0)
hours_per_week = st.sidebar.slider("Hours Per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", data['native.country'].unique())

# Assemble input
st.markdown("## 🧾 Your Selected Inputs")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style='background-color:#e9ecef; color:#212529; padding:15px 25px; border-radius:10px;
                 box-shadow:0 2px 8px rgba(0,0,0,0.1); font-size:15px; line-height:1.6'>
    <b>Age:</b> {age}<br>
    <b>Workclass:</b> {workclass}<br>
    <b>Fnlwgt:</b> {fnlwgt}<br>
    <b>Education:</b> {education}<br>
    <b>Education Num:</b> {education_num}<br>
    <b>Marital Status:</b> {marital_status}<br>
    <b>Occupation:</b> {occupation}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background-color:#e9ecef; color:#212529; padding:15px 25px; border-radius:10px;
                 box-shadow:0 2px 8px rgba(0,0,0,0.1); font-size:15px; line-height:1.6'>
    <b>Relationship:</b> {relationship}<br>
    <b>Race:</b> {race}<br>
    <b>Gender:</b> {gender}<br>
    <b>Capital Gain:</b> {capital_gain}<br>
    <b>Capital Loss:</b> {capital_loss}<br>
    <b>Hours per Week:</b> {hours_per_week}<br>
    <b>Native Country:</b> {native_country}
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")
if st.button("🚀 Submit to Predict"):
    input_dict = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education.num': education_num,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week,
        'native.country': native_country
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = preprocess(input_df)
    prediction = model.predict(input_encoded)

    st.markdown("## 🎯 Prediction Result")
    if prediction[0] == 1:
        st.success("✅ The model predicts: Salary >50K")
    else:
        st.warning("🔻 The model predicts: Salary <=50K")
st.markdown(
    """
    <hr style="margin-top: 50px; border: 1px solid #ccc;">
    <div style='text-align: center; color: white; font-size: 14px;'>
        🚀 Made by 🧑🏻‍🎓<b>K JASHUVA AKHIL</b>
    </div>
    """,
    unsafe_allow_html=True
)
