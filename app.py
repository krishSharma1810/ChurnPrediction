import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# loading the train model
model1= tf.keras.models.load_model('model.h5')

# load the encoders and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file :
    scaler = pickle.load(file)

# streamlit app 
st.title('Customer Churn Prediction')


# inputs from user
geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender= st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products= st.slider('Number of Products',1,4)
has_cr_card= st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])


# prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})


# one-hot encoding the input
geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))


# combine with the input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


# scale the data
input_data_scaled=scaler.transform(input_data)


# predict the data
prediction=model1.predict(input_data_scaled)
prediction_praba=prediction[0][0]

st.write(f'Chrun Probability: {prediction_praba:.2f}')

if prediction_praba > 0.5:
    st.write('The Customer is likely to Churn')
else:
    st.write('The Customer is not likely to Churn')
