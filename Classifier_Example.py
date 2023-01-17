import pandas as pd
import numpy as np
import xgboost as xgb
#import sklearn

def read_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['Class'] = df['Class'].str.replace("'", '').astype(int)
    return df

def train_classifier(data: np.ndarray, labels: np.ndarray,train_test_split:float=0.8) -> xgb.XGBClassifier:
    n_rows = len(labels)
    n_training_rows = int(n_rows*train_test_split)
    train_data, test_data = data[0:n_training_rows,:], data[n_training_rows,:]
    train_labels, test_labels = data[0:n_training_rows], data[n_training_rows:]
    train_test_split

    model = xgb.XGBClassifier()
    model.fit(data, labels, verbose=True)
    model_labels = model.predict(test_data)
    print('fraction correct', np.sum(model_labels == test_labels)/len(test_labels))

    return model

if __name__ == '__main__':
    df = read_data('/Users/zachstrennen/Downloads/creditcard_csv.csv')

    data = df[['V'+str(i) for i in range(1, 29)]].values
    labels = df['Class'].values
    model = train_classifier(data, labels)
