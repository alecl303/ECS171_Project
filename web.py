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


#Import dataset
dataset = pd.read_csv("/Users/prashansagoel/Desktop/ECS171_Project/fake_job_postings.csv")

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
    st.write("Attributes")

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
            first_4 = requirements_options[:4]
            requirements_options = requirements_options[4:] + first_4
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


encoder = preprocessing.OrdinalEncoder()
data = encoder.fit_transform(dataset)
#Encode all strings into numbers
encoded_dataset = pd.DataFrame(data, columns=dataset.columns)

#Specify x and y data
OldX = encoded_dataset.drop(Class, axis = 1)
Oldy = encoded_dataset[Class]

#Oversample to get similar amounts in x and y
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(OldX,Oldy)

# normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
X_rescaled = scaler.fit_transform(X)
X = pd.DataFrame(data = X_rescaled, columns = X.columns)

set_of_classes = y.value_counts().index.tolist()
set_of_classes= pd.DataFrame({Class: set_of_classes})

#splitting data into ratio 80:20
data_train, _, class_train, _ = train_test_split(X, y, test_size=0.2)

#Setup model with select hyperparams
mlp = MLPClassifier(solver = 'sgd', random_state = 42, activation = 'logistic', learning_rate_init = 0.2, batch_size = 100, hidden_layer_sizes = (10,2), max_iter = 6000)

#Fit the model to data
mlp.fit(data_train, class_train)

data = [title,location,requirements,telecommuting, logo, questions, employment, experience, education, industry, function]
#pred = mlp.predict([data])

if submit_button:
    st.write(f"Model 1: {pred}")

