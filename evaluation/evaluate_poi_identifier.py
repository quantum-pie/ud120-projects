#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
### it's all yours from here forward!
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)


def classify(features_train, labels_train):
    ### your code goes here!
    clf = tree.DecisionTreeClassifier()
    return clf.fit(features_train, labels_train)


classifier = classify(features_train, labels_train)
predicted_labels = classifier.predict(features_test)
print("Detected POIs: " + str(sum(predicted_labels)))
print("Total people: " + str(len(predicted_labels)))

for true_label, pred_label in zip(labels_test, predicted_labels):
    print("True label: " + str(true_label) + " Predicted label: " + str(pred_label))

print(accuracy_score(labels_test, np.zeros(len(predicted_labels))))
print(accuracy_score(labels_test, predicted_labels))

print("Precision: " + str(precision_score(labels_test, predicted_labels)))
print("Recall: " + str(recall_score(labels_test, predicted_labels)))

