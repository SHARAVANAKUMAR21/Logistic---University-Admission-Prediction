import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("admission.csv")
    return data

data = load_data()

# Streamlit interface
st.title('University Admission Prediction')

gre_score = st.number_input('Enter GRE Score', min_value=290, max_value=350, step=1)
gpa = st.number_input('Enter GPA', min_value=2.5, max_value=4.0, step=0.1)

st.write('GRE Score entered:', gre_score)
st.write('GPA entered:', gpa)

# Split the data into independent and dependent variables
X = data[['GRE Score', 'GPA']]
y = data['Admitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
def predict_admission(gre_score, gpa):
    prediction = model.predict([[gre_score, gpa]])
    return prediction[0]

if st.button('Predict Admission'):
    prediction = predict_admission(gre_score, gpa)
    st.write(f'The prediction result is: {prediction}')