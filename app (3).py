import pickle
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# load model
model = pickle.load(open('rf_model.pkl','rb'))
# title
st.title('Heart attack risk classification app')

# input variables
# Age	RestingBP	Cholesterol	FastingBS	MaxHR	Oldpeak
Age = st.number_input('Age',min_value=20,max_value=100,value=25)
RestingBP = st.number_input('RestingBP',min_value=0,max_value=300,value=100)
Cholesterol = st.number_input('Cholesterol',min_value=0,max_value=600,value=100)
MaxHR = st.number_input('MaxHR',min_value=60,max_value=600,value=150)
Oldpeak = st.number_input('Oldpeak',min_value=-3,max_value=10,value=2)
FastingBS = st.selectbox('FastingBS',(0,1))
gender = st.selectbox('Gender',('M','F'))
ChestPainType = st.selectbox('ChestPainType',('ATA', 'NAP' ,'ASY' ,'TA'))
RestingECG = st.selectbox('RestingECG',('Normal' ,'ST', 'LVH'))
ExerciseAngina = st.selectbox('ExerciseAngina',('N' ,'Y'))
ST_Slope = st.selectbox('ST_Slope',('Up' ,'Flat' ,'Down'))

# encoding
# Exercise_Angina: Label Encoder
Exercise_Angina = 1 if ExerciseAngina=='Y' else 0

#Sex_F	Sex_M get dummies
Sex_F = 1 if gender=='F' else 0
Sex_M = 1 if gender=='M' else 0

#ChestPainType,RestingECG,ST_Slope

ChestPainType_dict = {'ASY':3,'NAP':2,'ATA':1,'TA':0}
Chest_PainType=ChestPainType_dict[ChestPainType]

Resting_ECG_dict = {'Normal':0,'LVH':1,'ST':2}
RestingECG=Resting_ECG_dict[RestingECG]

ST_Slope_dict ={'Down':0,'Up':1,'Flat':2}
st_Slope = ST_Slope_dict[ST_Slope]

# create a dataframe
input_features=pd.DataFrame({'Age':[Age],'RestingBP':[RestingBP],
                      'Cholesterol':[Cholesterol],       
    'FastingBS':[FastingBS],'MaxHR':[MaxHR],'Oldpeak':[Oldpeak],
                         'Exercise_Angina':[Exercise_Angina],
                             'Sex_F':[Sex_F],'Sex_M':[Sex_M],
                             'Chest_PainType':[Chest_PainType],
                             'Resting_ECG':[RestingECG],
                             'st_Slope':[st_Slope]                                
})

# scaling
scaler = StandardScaler()
input_features[['Age','RestingBP', 'Cholesterol','MaxHR']]= scaler.fit_transform(input_features[['Age','RestingBP', 'Cholesterol','MaxHR']])

if st.button('Predict'):
  predictions = model.predict(input_features)
  if predictions==1:
    st.error('⚠️High risk of Heart Attack')
  else:
    st.success('😊Low risk of Heart Attack')