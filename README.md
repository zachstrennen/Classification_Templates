# Classification Templates using Crypto Scams
This repository contains base templates for different classification methods; specifically, XGBoost and Gaussian Naive Bayes. The dataset being tested on contains data on crypto scams and rug pulls from 2023 and earlier. In this example, the code is used to classify an instance as either a rug pull or a scam.

The code is split up into three modules. The python file crypto_classification contains all functions to be used. The python file classification_driver constains examples of how to use these functions. The python file test_classification contains base unit tests using pytest.

The code requires the python packages pandas, scikit-learn, xgboost, and pytest.
