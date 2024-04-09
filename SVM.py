# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:18:05 2024

@author: Win
"""

import data_exploration
from sklearn.pipeline import Pipeline


import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = data_exploration.data_load()

# Create SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

###########################################################################################
# Build SVM
###########################################################################################

# linear Kernel
clf_linear_ailin = SVC(kernel='linear', C=0.1)
clf_linear_ailin.fit(X_train, y_train)
# Predictions on the training set
y_train_pred_linear = clf_linear_ailin.predict(X_train)
# Predictions on the testing set
y_test_pred_linear = clf_linear_ailin.predict(X_test)


# Accuracy scores
train_accuracy_linear = accuracy_score(y_train, y_train_pred_linear)
test_accuracy_linear = accuracy_score(y_test, y_test_pred_linear)

print("Accuracy on training set (linear):", train_accuracy_linear)
print("Accuracy on testing set (linear):", test_accuracy_linear)

# Confusion matrix
conf_matrix_linear = confusion_matrix(y_test, y_test_pred_linear)
print("Confusion matrix (linear):")
print(conf_matrix_linear)
from sklearn.metrics import ConfusionMatrixDisplay
disp_linear = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_linear,display_labels=clf_linear_ailin.classes_)
disp_linear.plot()
#RBF Kernel
clf_rbf_ailin = SVC(kernel='rbf')
clf_rbf_ailin.fit(X_train, y_train)

# Predictions on the training set
y_train_pred_rbf = clf_rbf_ailin.predict(X_train)
# Predictions on the testing set
y_test_pred_rbf = clf_rbf_ailin.predict(X_test)

# Accuracy scores
train_accuracy_rbf = accuracy_score(y_train, y_train_pred_rbf)
test_accuracy_rbf = accuracy_score(y_test, y_test_pred_rbf)

print("Accuracy on training set (rbf):", train_accuracy_rbf)
print("Accuracy on testing set (rbf):", test_accuracy_rbf)

# Confusion matrix
conf_matrix_rbf = confusion_matrix(y_test, y_test_pred_rbf)
print("Confusion matrix (rbf):")
print(conf_matrix_rbf)
disp_rbf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rbf, display_labels=clf_rbf_ailin.classes_)
disp_rbf.plot()

#poly Kernel
clf_poly_ailin = SVC(kernel='poly')
clf_poly_ailin.fit(X_train, y_train)

# Predictions on the training set
y_train_pred_poly = clf_poly_ailin.predict(X_train)
# Predictions on the testing set
y_test_pred_poly = clf_poly_ailin.predict(X_test)

# Accuracy scores
train_accuracy_poly = accuracy_score(y_train, y_train_pred_poly)
test_accuracy_poly = accuracy_score(y_test, y_test_pred_poly)

print("Accuracy on training set (poly):", train_accuracy_poly)
print("Accuracy on testing set (poly):", test_accuracy_poly)

# Confusion matrix
conf_matrix_poly = confusion_matrix(y_test, y_test_pred_poly)
print("Confusion matrix (poly):")
print(conf_matrix_poly)
disp_poly = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_poly, display_labels=clf_poly_ailin.classes_)
disp_poly.plot()

#sigmoid Kernel
clf_sigmoid_ailin = SVC(kernel='sigmoid')
clf_sigmoid_ailin.fit(X_train, y_train)

# Predictions on the training set
y_train_pred_sigmoid = clf_sigmoid_ailin.predict(X_train)
# Predictions on the testing set
y_test_pred_sigmoid = clf_sigmoid_ailin.predict(X_test)

# Accuracy scores
train_accuracy_sigmoid = accuracy_score(y_train, y_train_pred_sigmoid)
test_accuracy_sigmoid = accuracy_score(y_test, y_test_pred_sigmoid)

print("Accuracy on training set (sigmoid):", train_accuracy_sigmoid)
print("Accuracy on testing set (sigmoid):", test_accuracy_sigmoid)

# Confusion matrix
conf_matrix_sigmoid = confusion_matrix(y_test, y_test_pred_sigmoid)
print("Confusion matrix (Sigmoid):")
print(conf_matrix_sigmoid)
disp_sigmoid = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_sigmoid, display_labels=clf_sigmoid_ailin.classes_)
disp_sigmoid.plot()




#################################################################################################
#use GridSearchCV to find best estimator
#################################################################################################
from sklearn.model_selection import GridSearchCV
svm_classifier = SVC( random_state=35)

clf = Pipeline(steps=[
        ('classifier', svm_classifier)
    ])
"""
param_grid = {
    'classifier__C': [1, 10, 100],
    'classifier__kernel': ['poly', 'rbf']
}
"""

param_grid = {
    'classifier__kernel': ['poly', 'rbf'],
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [0.01, 0.1, 1.0],
}

grid_search_ailin = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search_ailin.fit(X_train, y_train)

print("Best parameters:", grid_search_ailin.best_params_)

print("Best estimator:", grid_search_ailin.best_estimator_)

best_model_ailin = grid_search_ailin.best_estimator_
best_model_ailin.fit(X_train,y_train)
 
# Print out the accuracy score
train_accuracy = best_model_ailin.score(X_train, y_train)
test_accuracy = best_model_ailin.score(X_test, y_test)
print("Accuracy score on training data:", train_accuracy)
print("Accuracy score on testing data:", test_accuracy)


import joblib

joblib.dump(best_model_ailin, 'best_model_ailin_addGamma.joblib')
joblib.dump(grid_search_ailin, 'full_pipeline_ailin_addGamma.joblib')
