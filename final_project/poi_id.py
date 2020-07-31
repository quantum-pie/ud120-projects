#!/usr/bin/python

import sys
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit, updateFeaturesInDict
from tester import dump_classifier_and_data


def createNewFeatures(features, features_list):
    features_list.append('from_poi_ratio')
    features_list.append('to_poi_ratio')

    # subtract 1 because features_list contains labels
    from_poi_to_this_person_idx = features_list.index('from_poi_to_this_person') - 1
    to_messages_idx = features_list.index('to_messages') - 1
    from_this_person_to_poi_idx = features_list.index('from_this_person_to_poi') - 1
    from_messages_idx = features_list.index('from_messages') - 1

    new_features = []
    for f in features:
        if f[to_messages_idx] != 0:
            # from_poi_to_this_person / to_messages
            from_poi_ratio = f[from_poi_to_this_person_idx] / float(f[to_messages_idx])
        else:
            from_poi_ratio = 0

        if f[from_messages_idx] != 0:
            # from_this_person_to_poi / from_messages
            to_poi_ratio = f[from_this_person_to_poi_idx] / float(f[from_messages_idx])
        else:
            to_poi_ratio = 0

        # add new features to the end of feature vector
        new_features.append(np.append(f, (from_poi_ratio, to_poi_ratio)))
    return new_features


def scaleFeatures(features):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(features)
    return scaler.transform(features)


def selectBestFeatures(features, features_list):
    from sklearn.feature_selection import SelectKBest

    selector = SelectKBest(k=5)
    selector.fit_transform(features, labels)
    selected_idx = selector.get_support()

    new_features = [f[selected_idx] for f in features]
    features_list[1:] = [i for (i, v) in zip(features_list[1:], selected_idx) if v]

    return new_features


def selectAndTrainClassifier(features, labels):
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold

    ### Using cross validation to select model through grid hyperparameter search
    ### Classifier is a pipeline of PCA dimensionality reduction and SVM
    ### Criteria for parameter search is F1-score: it reflects requirement
    ### of maximizing both precision and recall

    ### Precision and recall of the best estimator are: 0.44 and 0.73

    estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    pipe = Pipeline(estimators)
    pipe.set_params(clf__kernel='rbf')
    pipe.set_params(clf__class_weight='balanced')
    pipe.set_params(clf__random_state=42)

    param_grid = dict(reduce_dim__n_components=[1, 2, 3, 4, 5],
                      clf__C=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                      clf__gamma=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    clf = GridSearchCV(estimator=pipe,
                       param_grid=param_grid,
                       cv=StratifiedKFold(10, random_state=42),
                       scoring=('f1'))

    clf = clf.fit(features, labels)
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    return clf.best_estimator_


def evaluateClassifier(clf):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedShuffleSplit

    cv_results = cross_validate(clf, features, labels,
                                cv=StratifiedShuffleSplit(n_splits=1000, random_state=42),
                                scoring=('precision', 'recall'))

    print("CV precision: " + str(np.mean(cv_results['test_precision'])))
    print("CV recall: " + str(np.mean(cv_results['test_recall'])))


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Just start with all the features (excluding e-mail address because it is basically a person label)
features_list = ['poi', 'salary', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock',
                 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'loan_advances', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
                 'to_messages', 'from_this_person_to_poi', 'from_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Remove TOTAL - outlier
my_dataset = data_dict
my_dataset.pop("TOTAL", 0)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 3: Create new feature(s)
### New features will be fractions of emails from/to this person to/from poi
### relative to all emails from/to this person
features = createNewFeatures(features, features_list)

### Scale features to [0, 1] range
features = scaleFeatures(features)

### Select 5 best features based on F-test -
### reduces amount of noise in the data due to features uncorrelated with labels
features = selectBestFeatures(features, features_list)

### Reflect changes in my_dataset
updateFeaturesInDict(my_dataset, features_list[1:], features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
clf = selectAndTrainClassifier(features, labels)

### Evaluate classifier
evaluateClassifier(clf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
