import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
credited_card_df = pd.read_csv(r'C:\Users\Ramadevi\OneDrive\Desktop\python\vs\creditcard.csv')

legit = credited_card_df[credited_card_df['Class'] == 0]
fraud = credited_card_df[credited_card_df['Class'] == 1]

legit_sample = legit.sample(n=492, random_state=2)

new_df = pd.concat([legit_sample, fraud], axis=0)

X = new_df.drop(columns='Class')
Y = new_df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

st.title('Credit Card Fraud Detection')

input_df = st.text_input(
    'Enter 30 values (V1, V2, ..., V28, Amount, Time) separated by commas:'
)

submit = st.button("Submit")

if submit:
    try:
        # FIXED variable name + conversion
        input_data = [float(x) for x in input_df.split(',')]
        input_data = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_data)

        if prediction[0] == 0:
            st.success("The transaction is Legitimate.")
        else:
            st.error("The transaction is Fraudulent.")

    except:
        st.warning("⚠️ Please enter valid numeric values separated by commas.")
