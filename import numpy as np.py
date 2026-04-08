import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# ✅ Load dataset (USE FULL PATH if needed)
credited_card_df = pd.read_csv(r"C:\Users\Ramadevi\OneDrive\Desktop\creditcard.csv")

# Basic info
st.write("Dataset Info")
st.write(credited_card_df.head())
st.write(credited_card_df.info())
st.write(credited_card_df.shape)

# Class distribution
st.write("Class Distribution:")
st.write(credited_card_df['Class'].value_counts())

# Separate legit & fraud
legit = credited_card_df[credited_card_df.Class == 0]
fraud = credited_card_df[credited_card_df.Class == 1]

# Sampling
legit_sample = legit.sample(n=492, random_state=2)
new_df = pd.concat([legit_sample, fraud], axis=0)

# Features & Labels
X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

st.write(f"Training Accuracy: {train_acc}")
st.write(f"Testing Accuracy: {test_acc}")

# ---------------- STREAMLIT APP ---------------- #

st.title('💳 Credit Card Fraud Detection')

st.write("Enter 30 values (Time, V1–V28, Amount) separated by commas:")

user_input = st.text_input("Input:")

if st.button("Predict"):
    try:
        input_list = user_input.split(',')
        
        # Convert to float
        input_data = np.array([float(i) for i in input_list]).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_data)

        if prediction[0] == 0:
            st.success("✅ The transaction is Legitimate.")
        else:
            st.error("🚨 The transaction is Fraudulent.")

    except:
        st.warning("⚠️ Please enter exactly 30 numeric values separated by commas.")