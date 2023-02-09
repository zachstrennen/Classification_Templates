import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sklearn.naive_bayes

def read_data(path:str) -> pd.DataFrame:
    """
    Read in the dataset from a selected directory.
    Adjust dataframe to specific format.
    :param path: String containing the path name.
    :return: Adjusted dataset pulled from directory path.
    """
    # Read in data from path
    df = pd.read_csv(path)

    # Convert column to float
    df["Amount Stolen"] = df["Amount Stolen"].replace(',', '', regex=True)
    df["Amount Stolen"] = df["Amount Stolen"].astype(float)

    # Convert string to date data type
    df['Date'] = pd.to_datetime(df['Date'], format='%b, %Y')

    # Only use years as a predictor (in terms of date)
    df['Date'] = pd.DatetimeIndex(df['Date']).year

    # Randomly shuffle the rows (for data splitting purposes)
    df = df.sample(frac=1)

    # Company name will have no effect on prediction
    df = df.drop("Company Name", axis=1)

    # Split categories into binary data (one hot encoding)
    df = pd.get_dummies(df)

    # Drop non-determinant feature
    df = df.drop("Rug Pull vs Scam_Scam", axis=1)

    return df

def split_data(df:pd.DataFrame,ratio:float,target:str) -> (pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame):
    """
    Split a dataframe by target and predictor variables.
    Decide which data will be test data and which data will be training data.
    :param df: Dataframe to be split.
    :param ratio: Ratio (float) of split between training and testing.
    :param target: String of target column name.
    :return: 4 dataframes - the train data for X and y, the test data for X and y.
    """
    X = df[df.columns[~df.columns.isin([target])]]
    y = df[[target]]
    train_X,test_X, train_y,test_y = train_test_split(X,y,test_size=ratio)
    return train_X,test_X, train_y,test_y

def train_data_XGB(train_X:pd.DataFrame,train_y:pd.DataFrame):
    """
    Build the model using XGBoost.
    Print out the accuracy of the model.
    Return the model.
    :param train_X: Train data for X.
    :param train_y: Train data for y.
    :return: Prediction model.
    """
    model = XGBClassifier()
    model.fit(train_X, train_y)
    return model

def train_data_bayes(train_X:pd.DataFrame,train_y:pd.DataFrame):
    """
    Build the model using Naive Bayes.
    Print out the accuracy of the model.
    Return the model.
    :param train_X: Train data for X.
    :param train_y: Train data for y.
    :return: Prediction model.
    """
    model = sklearn.naive_bayes.GaussianNB()
    model.fit(train_X, train_y.values.ravel(),sample_weight=None)
    return model

def possess(model,test_X:pd.DataFrame,test_y:pd.DataFrame):
    """
    Take in the fitted model and use test data to output the models accuracy.
    :param model: Already built model.
    :param test_X: train data for X.
    :param test_y: train data for y.
    :return:
    """
    y_pred = model.predict(test_X)
    predictions = [round(value) for value in y_pred]
    # Compare for accuracy
    accuracy = accuracy_score(test_y, predictions)
    return accuracy

def predict(model, df:pd.DataFrame):
    """
    Take in the model and a dataset of value to use for prediction.
    Use the model on the dataset and return a binary vector of predictions.
    :param model: Already built model.
    :param df: Dataframe specific to the models features.
    :return: Binary vector of predictions.
    """
    # Fit the model to the dataset that will be used for prediction
    predictions = model.predict(df)
    predictions = [round(value) for value in predictions]
    # Return a binary vector of predictions
    return predictions


if __name__ == '__main__':

    # Read in data
    df = read_data("/Users/zachstrennen/Downloads/NFT Rug Pulls.csv")

    # Predict if Rug Pull using two separate models
    train_X,test_X, train_y,test_y = split_data(df,0.5,"Rug Pull vs Scam_Rug Pull")
    XGB_model = train_data_XGB(train_X,train_y)
    bayes_model = train_data_bayes(train_X, train_y)

    # Find the accuracy of both models and print it
    accuracy = possess(XGB_model,test_X,test_y)
    print("XGBOOST MODEL")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("")
    accuracy = possess(bayes_model, test_X, test_y)
    print("BAYES MODEL")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))











