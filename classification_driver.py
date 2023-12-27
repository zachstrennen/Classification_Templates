from crypto_classification import read_data,\
    split_data, train_data_XGB, train_data_bayes, possess

if __name__ == '__main__':

    # Read in data
    df = read_data("data/NFT Rug Pulls.csv")

    # Predict if Rug Pull using two separate models
    train_X, test_X, train_y, test_y = split_data(df, 0.5,
                                                  "Rug Pull vs Scam_Rug Pull")
    XGB_model = train_data_XGB(train_X, train_y)
    bayes_model = train_data_bayes(train_X, train_y)

    # Find the accuracy of both models and print it
    accuracy = possess(XGB_model, test_X, test_y)
    print("XGBOOST MODEL")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("")
    accuracy = possess(bayes_model, test_X, test_y)
    print("BAYES MODEL")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
