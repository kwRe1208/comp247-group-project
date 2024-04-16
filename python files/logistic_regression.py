# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:54:27 2024

@author: Win
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

import data_exploration
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = data_exploration.data_load()

# Define the logistic regression model
#model = LogisticRegression() # ConvergenceWarning: lbfgs failed to converge (status=1):
model = LogisticRegression(solver='lbfgs', max_iter=1000)
# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('model', model)])

# the initial try 
clf.fit(X_train, y_train)
#Preprocessing of validation data, get predictions
y_test_preds = clf.predict(X_test)

# Get predictions for the training data
y_train_preds = clf.predict(X_train)

#Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Training Accuracy:', accuracy_score(y_train, y_train_preds))
print('Test Accuracy:', accuracy_score(y_test, y_test_preds))
print(confusion_matrix(y_test, y_test_preds))
print(classification_report(y_test, y_test_preds))


# Confusion matrix
conf_matrix= confusion_matrix(y_test, y_test_preds)
print("Confusion matrix:")
print(conf_matrix)
from sklearn.metrics import ConfusionMatrixDisplay
disp_cm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=clf.classes_)
disp_cm.plot()


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'model__C': [0.1, 1, 10, 100],  # Regularization parameter
    'model__solver': ['liblinear', 'lbfgs']  # Solver for optimization
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best estimator:", grid_search.best_estimator_)

best_model = grid_search.best_estimator_
best_model.fit(X_train,y_train)
 
# Print out the accuracy score
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
print("Accuracy score on training data:", train_accuracy)
print("Accuracy score on testing data:", test_accuracy)

# Save the best model to a file
joblib.dump(best_model, 'best_logistic_regression_model.pkl')


