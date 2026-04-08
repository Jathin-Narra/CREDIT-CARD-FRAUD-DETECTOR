# CREDIT-CARD-FRAUD-DETECTOR
a python project which has linear regression model  and the first time, i integrated the python code and ran it as a website with the help of streamlit
import numpy as np
import pandas as pd
pd.read_csv('creditcard.csv')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st


credited_card_df = pd.read_csv('creditcard.csv')
credited_card_df.head()
credited_card_df.info()
credited_card_df.shape
credited_card_df['Class'].value_counts()
legit = credited_card_df[credited_card_df.Class ==0]
fraud = credited_card_df[credited_card_df.Class ==1]
fraud['Class']
legit.Amount.describe()
fraud.Amount.describe()
credited_card_df.groupby('Class').mean()
legit_sample = legit.sample(n=492)
new_df = pd.concat([legit_sample,fraud],axis=0)
new_df
new_df['Class'].value_counts()
new_df.groupby('Class').mean()
X = new_df.drop(columns='Class',axis=1)
Y = new_df['Class']
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model=LogisticRegression()
model.fit(X_train, Y_train)
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)
# web app
st.title('Credit Card Fraud Detection')
input_df = st.text_input('Enter the transaction details (V1, V2, ..., V28, Amount):')
input_df_splitted = inputdf.split(',')
submit = st.button("submit")
if submit:
    input_data = np.array(input_df_splitted).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        st.write("The transaction is Legitimate.")
    else:
        st.write("The transaction is Fraudulent.")
