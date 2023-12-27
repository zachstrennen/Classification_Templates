import pytest
import pandas as pd
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from crypto_classification import read_data, split_data,\
    train_data_XGB, train_data_bayes


def test_read_data():
    # Add test data
    path = "data/NFT Rug Pulls.csv"
    df = read_data(path)

    # Add your assertions based on the expected output
    assert isinstance(df, pd.DataFrame)
    assert "Amount Stolen" in df.columns
    assert "Date" in df.columns
    assert "Company Name" not in df.columns


def test_split_data():
    # Create a sample dataframe for testing
    df = pd.DataFrame({'feature1': [1, 2, 3, 4],
                       'feature2': ['A', 'B', 'C', 'D'],
                       'target': [0, 1, 0, 1]})

    # Specify the target column and test ratio
    target_column = 'target'
    test_ratio = 0.2

    train_X, test_X, train_y, test_y =\
        split_data(df, test_ratio, target_column)

    # Add assertions based on the expected output
    assert isinstance(train_X, pd.DataFrame)
    assert isinstance(test_X, pd.DataFrame)
    assert isinstance(train_y, pd.DataFrame)
    assert isinstance(test_y, pd.DataFrame)


def test_train_data_XGB():
    # Create a minimal dataset for testing
    train_X = pd.DataFrame({'feature1': [1, 2, 3, 4],
                            'feature2': [5, 6, 7, 8]})
    train_y = pd.DataFrame({'target': [0, 1, 0, 1]})

    # Call the function to train the XGBoost model
    model = train_data_XGB(train_X, train_y)

    # Check if the model is an instance of XGBClassifier
    assert isinstance(model, XGBClassifier)


def test_train_data_bayes():
    # Create a minimal dataset for testing
    train_X = pd.DataFrame({'feature1': [1, 2, 3, 4],
                            'feature2': [5, 6, 7, 8]})
    train_y = pd.DataFrame({'target': [0, 1, 0, 1]})

    # Call the function to train the Naive Bayes model
    model = train_data_bayes(train_X, train_y)

    # Check if the model is an instance of GaussianNB
    assert isinstance(model, GaussianNB)
