'''
pip install joblib
import joblib

# Assuming you have trained a Random Forest Classifier named rf_classifier
# Save the trained model to a file
joblib.dump(rf_classifier, 'random_forest_model.pkl')

# Load the trained model from the file
rf_classifier = joblib.load('random_forest_model.pkl')


'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
# import seaborn as sns


# Step 2: Reading the dataset

dataset = pd.read_csv('keypoints.csv')
dataset = dataset.reset_index(drop=True)

dataset['key_35'] = pd.factorize(dataset['key_35'])[0]

X = dataset.iloc[:,:-2].values
y = dataset.iloc[:,34:35].values

#coversion of catgorical data to numerical data
# dataset['Category'] = pd.factorize(df['Category'])[0]

# print(f"x={X}")
# print(f"y={y}")
# print(dataset.isnull().sum())
# print(X.shape)

# ----------------Instantiating a LinearRegression Model----------------
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# model = LogisticRegression(C=1e5)
# model = LinearRegression()
'''
Applying random forest classifier
'''
rmse = 0
while rmse < 0.60:
    model = RandomForestClassifier(max_depth=2, random_state=0)
    X, y = make_classification(n_samples=54, n_features=34,n_informative=34, n_redundant=0,random_state=0, shuffle=True)

    # Splitting the datasets into training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
    print(f"xtrain= {X_train.shape}")
    print(f"X_test= {X_test.shape}")


# Fitting our model
    model.fit(X_train, y_train)

    # Running Evaluation Metrics
    from sklearn.metrics import mean_squared_error, r2_score
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=True)

    print('\n The r2 is:', r2)
    print('The rmse is: \n', rmse)


# if rmse > 0.60:
    # Save the trained model to a file
    # with open('./model/pose_model.pkl','wb'):
joblib.dump(model, 'pose_model1.pkl')
print(f"-------------model saved-----------------")
