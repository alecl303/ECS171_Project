import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


#Import dataset
dataset = pd.read_csv("fake_job_postings.csv")

#Chose class to predict
Class = "fraudulent"

#Fill empty rows with unspecified
df = dataset['requirements'].fillna('Unspecified')

#Combine datasets
dataset = pd.merge(dataset,df)

#Remove any row that has an empty column
dataset.dropna(axis='index', inplace=True)

#Remove select columns
dataset = dataset.drop(columns=['salary_range','company_profile','description','benefits','department','job_id'])

st.title("Fraudulent Job Posting Predictor")

with st.form("user_input_form"):
    st.write("**Attributes**")

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            title_options = np.unique(dataset['title'])
            title = st.selectbox("title", title_options)

        with col2:
            location_options = np.unique(dataset['location'])
            location = st.selectbox("location", location_options)

        with col3:
            requirements_options = np.unique(dataset['requirements'])
            requirements_options = list(requirements_options)
            first_16 = requirements_options[:17]
            requirements_options = requirements_options[17:] + first_16
            requirements = st.selectbox("requirements", requirements_options)

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
                telecommuting_options = np.unique(dataset['telecommuting'])
                telecommuting = st.selectbox("telecommuting", telecommuting_options)

        with col2:
            logo_options = np.unique(dataset['has_company_logo'])
            logo = st.selectbox("has_company_logo", logo_options)

        with col3:
            questions_options = np.unique(dataset['has_questions'])
            questions = st.selectbox("has_questions", questions_options)
    
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
                employment_options = np.unique(dataset['employment_type'])
                employment = st.selectbox("employment_type", employment_options)

        with col2:
            experience_options = np.unique(dataset['required_experience'])
            experience = st.selectbox("required_experience", experience_options)

        with col3:
            education_options = np.unique(dataset['required_education'])
            education = st.selectbox("required_education", education_options)
    
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
                industry_options = np.unique(dataset['industry'])
                industry = st.selectbox("industry", industry_options)

        with col2:
            function_options = np.unique(dataset['function'])
            function = st.selectbox("function", function_options)
    
    col1 = st.columns(1)
    with col1[0]:
        submit_button = st.form_submit_button(label='Predict')

#Encode all strings into numbers
encoder = preprocessing.OrdinalEncoder()
data = encoder.fit_transform(dataset)
encoded_dataset = pd.DataFrame(data, columns=dataset.columns)

title_dictionary = {}
location_dictionary = {}
requirements_dictionary = {}
telecommuting_dictionary = {}
logo_dictionary = {} 
questions_dictionary = {}
employment_dictionary = {}
experience_dictionary = {}
education_dictionary = {}
industry_dictionary = {}
function_dictionary = {}

#Create mappings between original input and ordinal encodings for each attribute
for (index1, row1), (index2, row2) in zip(encoded_dataset.iterrows(), dataset.iterrows()):
    key = (dataset.loc[index2])['title']
    val = (encoded_dataset.loc[index1])['title']
    if key not in title_dictionary:
        title_dictionary[key] = val
    
    key = (dataset.loc[index2])['location']
    val = (encoded_dataset.loc[index1])['location']
    if key not in location_dictionary:
        location_dictionary[key] = val

    key = (dataset.loc[index2])['requirements']
    val = (encoded_dataset.loc[index1])['requirements']
    if key not in requirements_dictionary:
        requirements_dictionary[key] = val
    
    key = (dataset.loc[index2])['telecommuting']
    val = (encoded_dataset.loc[index1])['telecommuting']
    if key not in telecommuting_dictionary:
        telecommuting_dictionary[key] = val
    
    key = (dataset.loc[index2])['has_company_logo']
    val = (encoded_dataset.loc[index1])['has_company_logo']
    if key not in logo_dictionary:
        logo_dictionary[key] = val

    key = (dataset.loc[index2])['has_questions']
    val = (encoded_dataset.loc[index1])['has_questions']
    if key not in questions_dictionary:
        questions_dictionary[key] = val
    
    key = (dataset.loc[index2])['employment_type']
    val = (encoded_dataset.loc[index1])['employment_type']
    if key not in employment_dictionary:
        employment_dictionary[key] = val

    key = (dataset.loc[index2])['required_experience']
    val = (encoded_dataset.loc[index1])['required_experience']
    if key not in experience_dictionary:
        experience_dictionary[key] = val
    
    key = (dataset.loc[index2])['required_education']
    val = (encoded_dataset.loc[index1])['required_education']
    if key not in education_dictionary:
        education_dictionary[key] = val

    key = (dataset.loc[index2])['industry']
    val = (encoded_dataset.loc[index1])['industry']
    if key not in industry_dictionary:
        industry_dictionary[key] = val
    
    key = (dataset.loc[index2])['function']
    val = (encoded_dataset.loc[index1])['function']
    if key not in function_dictionary:
        function_dictionary[key] = val

#Specify x and y data
OldX = encoded_dataset.drop(Class, axis = 1)
Oldy = encoded_dataset[Class]

#Oversample to get similar amounts in x and y
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(OldX,Oldy)

#Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X)
X = pd.DataFrame(data = X_rescaled, columns = X.columns)

#splitting data into ratio 80:20
data_train, data_test, class_train, class_test = train_test_split(X, y, test_size=0.2)

#Setup model with select hyperparams found from gridsearch
mlp = MLPClassifier(activation='logistic', alpha=0.001, hidden_layer_sizes=(100, 100), learning_rate_init=0.15, max_iter=800)

#Fit the model to data
mlp.fit(data_train, class_train)

#Build second model based on rbf 
svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(data_train, np.asarray(class_train))

#Convert textual inputs into ordinal encodings to use with model
title = title_dictionary[title]
location = location_dictionary[location]
requirements = requirements_dictionary[requirements]
employment = employment_dictionary[employment]
experience = experience_dictionary[experience]
education = education_dictionary[education]
industry = industry_dictionary[industry]
function = function_dictionary[function]

data = [title,location,requirements,telecommuting, logo, questions, employment, experience, education, industry, function]

#Make predictions 
pred = mlp.predict([data])
pred2 = svc_rbf.predict([data])

if submit_button:
    if pred == [0.]:
        st.write(f"MLP Classifier: 0 (Not Fraudulent)")
    else:
        st.write(f"MLP Classifier: 1 (Fraudulent)")
    
    if pred2 == [0.]:
        st.write(f"SVM Classifier: 0 (Not Fraudulent)")
    else:
        st.write(f"SVM Classifier: 1 (Fraudulent)")
    

