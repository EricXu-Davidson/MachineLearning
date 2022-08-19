import pandas as pd
import numpy as np
import sklearn
from sklearn import svm, metrics, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


#read in data
ori_data = pd.read_csv("data/compiled_data.csv")

#shuffle data
shuf_data = shuffle(ori_data)

#store values in the "cancer" column
y_value = shuf_data["cancer"]

y_drop = shuf_data.drop(columns=["cancer"])
scaler = MinMaxScaler()
result = scaler.fit(y_drop)

#normalized data is a 2d array
nor_data = scaler.transform(y_drop)


#split data
X_train, x_test, Y_train, y_test = train_test_split(nor_data, y_value, test_size = 0.20, random_state= 42)
x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train, test_size= 1/8, random_state= 42)

#hyperparameter tuning
parameters = {'kernel':('linear','poly'), 'C':[1,2,3,4,5,6,7,8,9,10]}
svc = svm.SVC()
gs = GridSearchCV(svc, parameters)
gs.fit(x_val, y_val)
# print(gs.cv_results_["params"])
# print(gs.cv_results_["rank_test_score"])
# print(gs.cv_results_["mean_test_score"])

#K_fold validation for poly level of 1,2,3
K_fold = svm.SVC(kernel = 'linear')
validation_result = cross_val_score(K_fold, x_val, y_val, cv = 5)
# print(validation_result.mean())

#svm model building and testing
clf = svm.SVC(kernel = 'linear', C = 1)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
# print(
#     f"Classification report for classifier {clf}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )
