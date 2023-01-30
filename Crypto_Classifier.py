import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def read_data(path:str) -> pd.DataFrame:
    """
    Read in the dataset from a selected directory.
    :param path: String containing the path name.
    :return: Dataset pulled from directory path.
    """
    df = pd.read_csv(path)
    return df

def convert_to_float(df:pd.DataFrame,column_name:str) -> pd.DataFrame:
    """
    Convert the numbers from string data types to float data types.
    :param path: Dataframe and the string name of the column that will be converted.
    :return: Dataframe with the updated column.
    """
    df[column_name] = df[column_name].replace(',','',regex=True)
    df[column_name] = df[column_name].astype(float)
    return df

def convert_date_to_year(df:pd.DataFrame,column_name:str) -> pd.DataFrame:
    """
    Convert the dates from string data types to int years.
    :param path: A dataframe and the string name of the column that will be converted
    :return: Dataframe with the updated column.
    """
    df[column_name] = pd.to_datetime(df[column_name], format='%b, %Y')
    df[column_name] = pd.DatetimeIndex(df['Date']).year
    return df

def remove_column(df:pd.DataFrame,column:str) -> (pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame):
    """
    Remove an unwanted column from a dataframe.
    :param path: Dataframe and the string of the column name to be removed.
    :return: Dataframe with the column removed.
    """
    df = df.drop(column, axis=1)
    return df

def shuffle_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Randomly shuffle the rows of a dataframe.
    :param path: A dataframe and to be shuffled.
    :return: Shuffled dataframe.
    """
    df = df.sample(frac=1)
    return df

def split_data(df:pd.DataFrame,ratio:float,target:str) -> (pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame):
    """
    Split a dataframe by target and predictor variables.
    Decide which data will be test data and which data will be training data.
    :param path: Dataframe to be split, ratio (float) of split between training and testing,
                 string of target column name.
    :return: 4 dataframes - the train data for X and y, the test data for X and y.
    """
    X = df[df.columns[~df.columns.isin([target])]]
    y = df[[target]]
    train_X,test_X, train_y,test_y = train_test_split(X,y,test_size=ratio)
    return train_X,test_X, train_y,test_y

def build_model(train_X:pd.DataFrame,train_y:pd.DataFrame,test_X:pd.DataFrame,test_y:pd.DataFrame):
    """
    Build the model using XGBoost.
    Print out the accuracy of the model.
    Return the model.
    :param path: 4 dataframes: test data for X and y, train data for X and y.
    :return: Prediction model.
    """
    model = XGBClassifier()
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    predictions = [round(value) for value in y_pred]
    # compare for accuracy
    accuracy = accuracy_score(test_y, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return model

if __name__ == '__main__':

    # Read in data
    df = read_data("/Users/zachstrennen/Downloads/NFT Rug Pulls.csv")
    df = convert_to_float(df, "Amount Stolen")

    # Only use years as a predictor (in terms of date)
    df = convert_date_to_year(df, "Date")

    # Randomly shuffle the rows (for data splitting purposes)
    df = shuffle_data(df)

    # Company name will have no effect on prediction
    df = remove_column(df,"Company Name")

    # Split categories into binary data (one hot encoding)
    df = pd.get_dummies(df)

    #Predict if Rug Pull
    df = remove_column(df, "Rug Pull vs Scam_Scam")
    train_X,test_X, train_y,test_y = split_data(df,0.5,"Rug Pull vs Scam_Rug Pull")
    model = build_model(train_X,train_y,test_X,test_y)






