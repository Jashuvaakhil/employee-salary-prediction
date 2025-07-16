
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

# Convert to DataFrame and encode
input_df = pd.DataFrame([input_dict])
input_encoded = preprocess(input_df)

# Predict
prediction = model.predict(input_encoded)

# Display result
if prediction[0] == 1:
    st.success("The model predicts: Salary >50K")
else:
    st.warning("The model predicts: Salary <=50K")
