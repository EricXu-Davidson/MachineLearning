import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import graphviz

data = pd.read_csv("data/compiled_data.csv")

#saves the features and classes for showing the graphs
features = data.columns
features = features.drop('cancer')
classes = ['Breast Invasive Carcinoma', 'Kidney Renal Clear Cell Carcinoma', 'Lung Adenocarcinoma',
            'Lung Squamous Cell Carcinoma', 'Pancreatic Adenocarcinoma', 'Uveal Melanoma']
shuffled_data = data.sample(frac=1)
shuffled_data = np.array(shuffled_data)
X = shuffled_data[:, :-1]
y = shuffled_data[:,-1]
scaler = MinMaxScaler()
result = scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# clf = tree.DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# print(accuracy_score(y_test,y_predict))

print("-------------- running models --------------")
for i in range(1, 100):
    print("-------------- new model --------------")
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("Accuracy      | ", accuracy_score(y_test, y_predict))
    print("Tree Depth    | ", i)
    dot_data = tree.export_graphviz(clf, out_file=None, 
                        feature_names=features,  
                        class_names=classes,  
                        filled=True, rounded=True,  
                        special_characters=True) 
    print("-------------- creating tree visualizations --------------")
    graph = graphviz.Source(dot_data)  
    graph.render('tree_model ' + str(i), format='png')

