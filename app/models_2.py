"""
models.py
database schemas for Kickstarter app
"""
# *** IMPORTS ***
from os import getenv
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

import requests
import urllib.parse

import pickle
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor



# Create a DB Object
DB = SQLAlchemy()






# *** ML MODEL FUNCTIONS ***


def build_preprocessor():
    """

    :return:
    """
    # Create categorical pipeline
    cat_pipe = Pipeline([
        ('encoder', OneHotEncoder())
    ])

    # Create numerical pipeline
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Create text pipeline
    text_pipe = Pipeline([
        ('vect', TfidfVectorizer(stop_words='english', max_features=1000))
    ])

    categorical = ['category', 'subcategory']
    numerical = ['goal', 'campaign_duration', 'latitude', 'longitude']
    text = 'name_and_blurb_text'

    preprocessor = ColumnTransformer([('text', text_pipe, text),
                                      ('cat', cat_pipe, categorical),
                                      ('num', num_pipe, numerical)
                                      ])

    return preprocessor


def import_and_clean_data_old(submit_dict):
    """

    :return:
    """
    # Import test data
    # df = pd.read_sql('SELECT * FROM public.model10k;', con=engine)
    df = pd.read_csv('app/data/Kickstarter_Data_For_Model_10k.csv')

    # Combine text features into 1 column for model pipeline compatibility
    df['name_and_blurb_text'] = df['name'] + ' ' + df['blurb']

    # Drop original 2 text feature columns
    df = df.drop(columns=['name', 'blurb'], axis=1)

    # Rearrange columns
    # cols = list(df.columns.values)
    df = df[['name_and_blurb_text',
             'goal',
             'campaign_duration',
             'latitude',
             'longitude',
             'category',
             'subcategory',
             'outcome',
             'days_to_success'
             ]]

    # Add new record submitted by user
    df = df.append(submit_dict,
                   ignore_index=True
                   )

    """Stub value for submit_dict (from testing/development)
        {'name_and_blurb_text':
                  'Bound Printed Matter: The Life of JoeMisfit An Autobiography Follow & get to know music memorabilia collector JoeMisfit on his vivid & harrowing journey through life all the way up to present day.',
                  'goal': 10000.0,
                  'campaign_duration': 10.0,
                  'latitude': 40.037875,
                  'longitude': -76.305514,
                  'category': 'publishing',
                  'subcategory': 'nonfiction'},
    """


    df = df.drop(columns=['outcome', 'days_to_success'])

    preprocessor = build_preprocessor()

    # # Load pickled preprocessor
    # preprocessor_path = r'data/pickle_preprocessor.pkl'
    # with open(preprocessor_path, 'rb') as file:
    #     preprocessor = pickle.load(file)

    return preprocessor.fit_transform(df)


def process_record_old(submit_dict):
    """

    :return:
    """
    # Load model from pickle file
    path = r'app/data/pickle_model_10k.pkl'
    with open(path, 'rb') as file:
        model_knn = pickle.load(file)

    # Populate mock data
    X_transformed = import_and_clean_data(submit_dict)

    # Test on last record (recently appended)
    test_num = X_transformed.shape[0] - 1

    results = model_knn.kneighbors(X_transformed[test_num][:], n_neighbors=3,
                                   return_distance=False)

    prediction = model_knn.predict(X_transformed[test_num][:])
    # print(prediction)

    prob = model_knn.predict_proba(X_transformed[test_num][:])

    return str({'prediction': prediction,
               'probability': prob,
               'NearestNeighbors': results})



def import_and_clean_data(input_feature_list):
    """

    :return:
    """
    # Break out input feature list into variables
    name_and_blurb_text = input_feature_list[0]
    funding_goal = input_feature_list[1]
    campaign_duration = input_feature_list[2]
    latitude = input_feature_list[3]
    longitude = input_feature_list[4]
    category = input_feature_list[5]
    subcategory = input_feature_list[6]

    # Import test data
    path = r"app/data/Kickstarter_Data_For_Model_10k_v2.csv"
    df = pd.read_csv(path)

    # Combine text features into 1 column for model pipeline compatibility
    df["name_and_blurb_text"] = df["name"] + " " + df["blurb"]

    # Drop original 2 text feature columns
    df = df.drop(columns=["name", "blurb"], axis=1)

    # Rearrange columns
    # cols = list(df.columns.values)
    df = df[
        [
            "name_and_blurb_text",
            "goal",
            "campaign_duration",
            "latitude",
            "longitude",
            "category",
            "subcategory",
            "outcome",
            "days_to_success",
        ]
    ]

    df = df.drop(columns=["outcome", "days_to_success"])

    # Append user input to dataset
    user_input_df = pd.DataFrame(
        [{
            "name_and_blurb_text": name_and_blurb_text,
            "goal": funding_goal,
            "campaign_duration": campaign_duration,
            "latitude": latitude,
            "longitude": longitude,
            "category": category,
            "subcategory": subcategory,
        }]
    )

    df = df.append(user_input_df)

    # Create preprocessor and transform data
    preprocessor = build_preprocessor()
    preprocessor.fit_transform(df)

    # Return transformed version of user input (last record in dataframe)
    return preprocessor.transform(df.iloc[-1:])


def process_record(input_feature_list, model_path):
    """

    :return:
    """
    # Load model from pickle file
    path = r"app/data/pickle_model_10k.pkl"
    with open(path, "rb") as file:
        model_knn = pickle.load(file)

    # Populate user data
    user_data = import_and_clean_data(input_feature_list)

    prediction = model_knn.predict(user_data)
    neighbors = model_knn.kneighbors(user_data,
                                     n_neighbors=3, return_distance=False)
    prob = model_knn.predict_proba(user_data)

    pred = str(prediction)
    pred = int(pred[1])

    # Convert 1 or 0 into text
    if pred == 1:
        prediction = "Successful"
    elif pred == 0:
        prediction = "Unsuccessful"

    return {'pred': prediction,
            'prob': round((1 - prob[0][0])*100)}



def get_lat_value(location_text):
    try:
        url = (
                "https://nominatim.openstreetmap.org/search/"
                + urllib.parse.quote(location_text)
                + "?format=json"
        )
        response = requests.get(url).json()
        lat = response[0]["lat"]
        lat = float(lat)
        return lat
    except:
        return 0.0


def get_lng_value(location_text):
    try:
        url = (
                "https://nominatim.openstreetmap.org/search/"
                + urllib.parse.quote(location_text)
                + "?format=json"
        )
        response = requests.get(url).json()
        lng = response[0]["lon"]
        lng = float(lng)
        return lng
    except:
        return 0.0
