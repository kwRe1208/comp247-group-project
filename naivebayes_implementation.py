# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:47:41 2024

@author: Kirstin Ramos
"""

import pandas as pd
import numpy as np

dataset_path = r'KSI.csv'

df = pd.read_csv(dataset_path)

df.head(5)

print(df['ROAD_CLASS'].value_counts())
print(df['ROAD_CLASS'].isnull().sum())

df.info()

df.columns.values

#Since 5 rows are missing target values (ACCLASS), we will remove them
df = df.dropna(subset=['ACCLASS'])

#We will remove the columns that are not useful for our model
meaningless_columns = ['INDEX_', 'ACCNUM', 'YEAR', 'DATE', 'TIME', 'STREET1',
                        'STREET2', 'OFFSET', 'FATAL_NO', 'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140',
                        'ObjectId', 'WARDNUM', 'DIVISION']

too_much_missing = ['PEDTYPE','CYCACT', 'CYCLISTYPE', 'PEDACT', 'CYCCOND', 'MANOEUVER', 'INITDIR']

#We will remove the columns with duplicate information
# X and Y are the same as LONGITUDE and LATITUDE
# VEHTYPE, PEDCOND, DRIVCOND, IMPACTYPE, DRIVACT duplicated because there are categorical columns for the same information
duplicated_columns = ['X', 'Y', 'VEHTYPE', 'PEDCOND', 'DRIVCOND', 'IMPACTYPE','LOCCOORD', 'HOOD_140', 'DRIVACT']


#columns need to fill Nan values
binary_map = {np.nan: 'No'}
fill_nan_columns = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
                    'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV',
                    'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'INJURY']

#columns which contain categorical data
categorical_columns = ['LIGHT', 'INVAGE', 'RDSFCOND', 
                        'DISTRICT', 'ROAD_CLASS', 'TRAFFCTL', 
                        'ACCLOC', 'VISIBILITY','INVTYPE']

try_exclude = ['ROAD_CLASS', 'TRAFFCTL', 'ACCLOC']


# Create a copy of the dataframe
df_origin = df.copy()

#drop meaningless columns, duplicated columns and columns with too much missing values
df = df.drop(columns=meaningless_columns)
df = df.drop(columns=duplicated_columns)
df = df.drop(columns=too_much_missing)

#Simplfy the categorical data
# LIGHT
# Daylight                10385
# Dark                     3687
# Dark, artificial         3300
# Dusk                      240
# Dusk, artificial          219
# Daylight, artificial      141
# Dawn                      110
# Dawn, artificial          101
# Other                       6

#We will simplify the LIGHT column to Daylight, Dark, Dusk, Dawn, Other
light_map = {
    'Daylight': 'Daylight', 
    'Dark': 'Dark', 
    'Dark, artificial': 'Dark', 
    'Dusk': 'Dusk', 
    'Dusk, artificial': 'Dusk',         
    'Daylight, artificial': 'Daylight', 
    'Dawn': 'Dawn', 
    'Dawn, artificial': 'Dawn', 
    'Other': 'Other'
    }

df['LIGHT'] = df['LIGHT'].map(light_map)

# IMPACTYPE
# Pedestrian Collisions     7293
# Turning Movement          2792
# Cyclist Collisions        1795
# Rear End                  1746
# SMV Other                 1457
# Angle                     1283
# Approaching                928
# Sideswipe                  506
# Other                      195
# SMV Unattended Vehicle     190
# Name: count, dtype: int64
# Null Values: 4

#Pedestrian, Cyclist are representing IMPACTYPE column, so we will drop it

# INVAGE
# unknown     2609
# 20 to 24    1710
# 25 to 29    1638
# 30 to 34    1384
# 35 to 39    1311
# 50 to 54    1302
# 40 to 44    1274
# 45 to 49    1239
# 55 to 59    1098
# 60 to 64     877
# 15 to 19     852
# 65 to 69     681
# 70 to 74     529
# 75 to 79     434
# 80 to 84     336
# 10 to 14     249
# 85 to 89     212
# 5 to 9       199
# 0 to 4       177
# 90 to 94      63
# Over 95       15
# Name: count, dtype: int64
# Null Values: 0

#We will simplify the INVAGE column to 0 to 20, 20 to 40, 40 to 60, 60 to 80, over 80
invage_map = {
    'unknown': 'unknown',
    '20 to 24': '20 to 39',
    '25 to 29': '20 to 39',
    '30 to 34': '20 to 39',
    '35 to 39': '20 to 39',
    '50 to 54': '40 to 59',
    '40 to 44': '40 to 59',
    '45 to 49': '40 to 59',
    '55 to 59': '40 to 59',
    '60 to 64': '60 to 79',
    '15 to 19': '0 to 19',
    '65 to 69': '60 to 79',
    '70 to 74': '60 to 79',
    '75 to 79': '60 to 79',
    '80 to 84': 'over 79',
    '10 to 14': '0 to 19',
    '85 to 89': 'over 79',
    '5 to 9': '0 to 19',
    '0 to 4': '0 to 19',
    '90 to 94': 'over 79',
    'Over 95': 'over 79'
    }

df['INVAGE'] = df['INVAGE'].map(invage_map)

# RDSFCOND
# Dry                     14594
# Wet                      3021
# Loose Snow                169
# Other                     145
# Slush                     102
# Ice                        77
# Packed Snow                44
# Loose Sand or Gravel       11
# Spilled liquid              1
# Name: count, dtype: int64
# Null Values: 25

#We will simplify the RDSFCOND column to Dry, Wet, Snow, Ice, Other
rdsfcond_map = {
    'Dry': 'Dry',
    'Wet': 'Wet',
    'Loose Snow': 'Wet',
    'Other': 'Other',
    'Slush': 'Wet',
    'Ice': 'Wet',
    'Packed Snow': 'Wet',
    'Loose Sand or Gravel': 'Other',
    'Spilled liquid': 'Other'
    }

df['RDSFCOND'] = df['RDSFCOND'].map(rdsfcond_map)

# fill the missing values with other
df['RDSFCOND'] = df['RDSFCOND'].fillna('Other')

# DISTRICT
# Toronto and East York    6125
# Etobicoke York           4207
# Scarborough              4111
# North York               3637
# Toronto East York           4
# Name: count, dtype: int64
# Null Values: 105

# DISTRICT column has 105 missing values, we will fill them with the most frequent value
df['DISTRICT'] = df['DISTRICT'].fillna(df['DISTRICT'].mode()[0])

# DRIVACT
# Driving Properly                4221
# Failed to Yield Right of Way    1541
# Lost control                     975
# Improper Turn                    573
# Other                            504
# Disobeyed Traffic Control        475
# Following too Close              251
# Exceeding Speed Limit            246
# Speed too Fast For Condition     208
# Improper Lane Change             122
# Improper Passing                 112
# Wrong Way on One Way Road          9
# Speed too Slow                     4
# Name: count, dtype: int64
# Null Values: 8948

#Redlight, Speeding, Ag_Driv, Alcohol, Disability are representing DRIVACT column, so we will drop it

# INITDIR
# East       3259
# West       3197
# South      3106
# North      3066
# Unknown     510
# Name: count, dtype: int64
# Null Values: 5051

# INITDIR column has 5051 missing values, we will drop it

# ROAD_CLASS
# Major Arterial         12951
# Minor Arterial          2840
# Collector                996
# Local                    841
# Expressway               132
# Other                     25
# Laneway                   11
# Expressway Ramp            9
# Pending                    7
# Major Arterial Ramp        1
# Name: count, dtype: int64
# Null Values: 376

# Simplify the ROAD_CLASS column to Major Arterial, Minor Arterial, Collector, Local, Other
road_class_map = {
    'Major Arterial': 'Major Arterial',
    'Minor Arterial': 'Minor Arterial',
    'Collector': 'Collector',
    'Local': 'Local',
    'Expressway': 'Other',
    'Other': 'Other',
    'Laneway': 'Other',
    'Expressway Ramp': 'Other',
    'Pending': 'Other',
    'Major Arterial Ramp': 'Other'
    }

df['ROAD_CLASS'] = df['ROAD_CLASS'].map(road_class_map)

# Fill the missing values with Other
df['ROAD_CLASS'] = df['ROAD_CLASS'].fillna('Other')

# TRAFFCTL
# No Control              8788
# Traffic Signal          7635
# Stop Sign               1380
# Pedestrian Crossover     198
# Traffic Controller       108
# Yield Sign                21
# Streetcar (Stop for)      16
# Traffic Gate               5
# School Guard               2
# Police Control             2
# Name: count, dtype: int64
# Null Values: 34

# Simplyfy the TRAFFCTL column to No Control, Traffic Signal, Stop Sign, Other
traffctl_map = {
    'No Control': 'No Control',
    'Traffic Signal': 'Traffic Signal',
    'Stop Sign': 'Stop Sign',
    'Pedestrian Crossover': 'Other',
    'Traffic Controller': 'Other',
    'Yield Sign': 'Other',
    'Streetcar (Stop for)': 'Other',
    'Traffic Gate': 'Other',
    'School Guard': 'Other',
    'Police Control': 'Other'
    }

df['TRAFFCTL'] = df['TRAFFCTL'].map(traffctl_map)

# Fill the missing values with Other
df['TRAFFCTL'] = df['TRAFFCTL'].fillna('Other')

# ACCLOC
# At Intersection          8689
# Non Intersection         2420
# Intersection Related     1200
# At/Near Private Drive     379
# Overpass or Bridge         17
# Laneway                    14
# Private Driveway           13
# Underpass or Tunnel         6
# Trail                       1
# Name: count, dtype: int64
# Null Values: 5450

# Simplyfy the ACCLOC column to At Intersection, Non Intersection, Other
accloc_map = {
    'At Intersection': 'At Intersection',
    'Non Intersection': 'Non Intersection',
    'Intersection Related': 'At Intersection',
    'At/Near Private Drive': 'Other',
    'Overpass or Bridge': 'Other',
    'Laneway': 'Other',
    'Private Driveway': 'Other',
    'Underpass or Tunnel': 'Other',
    'Trail': 'Other'
    }   

df['ACCLOC'] = df['ACCLOC'].map(accloc_map)

# Fill the missing values with Other
df['ACCLOC'] = df['ACCLOC'].fillna('Other')

# VISIBILITY
# Clear                     15714
# Rain                       1879
# Snow                        351
# Other                        97
# Fog, Mist, Smoke, Dust       50
# Freezing Rain                47
# Drifting Snow                21
# Strong wind                  10
# Name: count, dtype: int64
# Null Values: 20

# Simplyfy the VISIBILITY column to Clear, Rain, Snow, Other

visibility_map = {
    'Clear': 'Clear',
    'Rain': 'Not Clear',
    'Snow': 'Not Clear',
    'Other': 'Other',
    'Fog, Mist, Smoke, Dust': 'Not Clear',
    'Freezing Rain': 'Not Clear',
    'Drifting Snow': 'Not Clear',
    'Strong wind': 'Other'
    }

df['VISIBILITY'] = df['VISIBILITY'].map(visibility_map)

# Fill the missing values with Other
df['VISIBILITY'] = df['VISIBILITY'].fillna('Other')

# INVTYPE
# Driver                  8274
# Pedestrian              3110
# Passenger               2766
# Vehicle Owner           1637
# Cyclist                  784
# Motorcycle Driver        697
# Truck Driver             346
# Other Property Owner     257
# Other                    186
# Motorcycle Passenger      39
# Moped Driver              30
# Driver - Not Hit          17
# Wheelchair                17
# In-Line Skater             5
# Cyclist Passenger          3
# Trailer Owner              2
# Pedestrian - Not Hit       1
# Witness                    1
# Moped Passenger            1
# Name: count, dtype: int64
# Null Values: 16

# Simplyfy the INVTYPE column to Driver, Pedestrian, Passenger, Vehicle Owner, Cyclist, Other

invtype_map = {
    'Driver': 'Driver',
    'Pedestrian': 'Pedestrian',
    'Passenger': 'Passenger',
    'Vehicle Owner': 'Vehicle Owner',
    'Cyclist': 'Cyclist',
    'Motorcycle Driver': 'Driver',
    'Truck Driver': 'Driver',
    'Other Property Owner': 'Other',
    'Other': 'Other',
    'Motorcycle Passenger': 'Passenger',
    'Moped Driver': 'Other',
    'Driver - Not Hit': 'Other',
    'Wheelchair': 'Other',
    'In-Line Skater': 'Other',
    'Cyclist Passenger': 'Passenger',
    'Trailer Owner': 'Vehicle Owner',
    'Pedestrian - Not Hit': 'Other',
    'Witness': 'Other',
    'Moped Passenger': 'Passenger'
    }

df['INVTYPE'] = df['INVTYPE'].map(invtype_map)

# Fill the missing values with Other
df['INVTYPE'] = df['INVTYPE'].fillna('Other')

# MANOEUVER
# Going Ahead                            6265
# Turning Left                           1786
# Stopped                                 620
# Turning Right                           476
# Slowing or Stopping                     282
# Changing Lanes                          216
# Parked                                  183
# Other                                   181
# Reversing                               122
# Unknown                                 122
# Making U Turn                           106
# Overtaking                               91
# Pulling Away from Shoulder or Curb       40
# Pulling Onto Shoulder or towardCurb      18
# Merging                                  18
# Disabled                                  4
# Name: count, dtype: int64
# Null Values: 7659

# Too difficult to simplify and too much missing values, we will drop it

print(df.columns.values)
print(len(df.columns.values))

# columns can try to exclude or include

#selected_categorical_columns = set(categorical_columns) - set(try_exclude)
selected_categorical_columns = categorical_columns
print(selected_categorical_columns)

# join selected_categorical_columns and fill_nan_columns
selected_cat_columns = list(selected_categorical_columns) + fill_nan_columns

print(selected_cat_columns)

df[fill_nan_columns] = df[fill_nan_columns].fillna(value='No')

df.info()

#export the cleaned dataset
df.to_csv(r'dataset\KSI_cleaned.csv', index=False)

data_map = {
    'Fatal': 1, 
    'Non-Fatal Injury': 0, 
    'Property Damage Only': 0
    }

nsa_count = df['HOOD_158'].value_counts().get('NSA', 0)
print(nsa_count)
df = df[df['HOOD_158'] != 'NSA']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Get transformed data
X_smote = df.drop(columns=['ACCLASS'], axis=1)
y_smote = df['ACCLASS'].map(data_map)

# Extracting numerical columns by excluding categorical ones
numerical_columns = list(set(X_smote.columns) - set(selected_cat_columns))
# Print the numerical columns
print("Numerical columns:", numerical_columns)

# Create the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), selected_cat_columns),
        ('num', ImbPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_columns)],
    remainder='passthrough'  # Pass through other columns as is
)

# Create the Naive Bayes pipeline
clf_no_smote = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')
# Assuming 'df_original' is your original DataFrame and 'df_modified' is after changes
X_no_smote = df.drop(columns=['ACCLASS'], axis=1)
y_no_smote = df['ACCLASS'].map(data_map)

# Split data
X_train_no_smote, X_test_no_smote, y_train_no_smote, y_test_no_smote = train_test_split(X_no_smote, y_no_smote, test_size=0.2, random_state=42)

# Train original model
#model_no_smote = GaussianNB()
clf_no_smote.fit(X_train_no_smote, y_train_no_smote)
y_pred_no_smote = clf_no_smote.predict(X_test_no_smote)
accuracy_no_smote = accuracy_score(y_test_no_smote, y_pred_no_smote)
print("Model with no SMOTE")
print(accuracy_no_smote)

print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')

# Create the Naive Bayes pipeline
clf = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=58)),
    ('classifier', GaussianNB())
])

# Define the k-fold cross-validation procedure
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Prepare lists to store results
acc_scores = []
recall_scores = []
precision_scores = []
f1_scores = []

# Enumerate through each fold
for train_ix, test_ix in kfold.split(X_smote, y_smote):
    # Split data
    X_train, X_test = X_smote.iloc[train_ix], X_smote.iloc[test_ix]
    y_train, y_test = y_smote.iloc[train_ix], y_smote.iloc[test_ix]
    
    # Define the modeling pipeline
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=58)),  # Applying SMOTE
        ('classifier', GaussianNB())
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Store scores
    acc_scores.append(accuracy_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))

# Calculate mean of each metric across all folds
print("Model with SMOTE")
print('Accuracy: %.3f' % mean(acc_scores))
print('Recall: %.3f' % mean(recall_scores))
print('Precision: %.3f' % mean(precision_scores))
print('F1 Score: %.3f' % mean(f1_scores))

print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')

from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score

# Define the k-fold cross-validation procedure
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Prepare lists to store results
acc_scores = []
recall_scores = []
precision_scores = []
f1_scores = []

# Function to find the threshold that maximizes a combination of precision and recall
def adjust_threshold(precisions, recalls, thresholds, beta=1):
    f_scores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls)
    return thresholds[np.argmax(f_scores)]

# Enumerate through each fold
for train_ix, test_ix in kfold.split(X_smote, y_smote):
    # Split data
    X_train, X_test = X_smote.iloc[train_ix], X_smote.iloc[test_ix]
    y_train, y_test = y_smote.iloc[train_ix], y_smote.iloc[test_ix]
    
    # Define the modeling pipeline
    model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=58)),  # Applying SMOTE
        ('classifier', GaussianNB())
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Get probabilities for the positive class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate precision-recall pairs for different probability thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, probabilities)

    # Get the optimal threshold using F0.5 Score which prioritizes precision
    optimal_threshold = adjust_threshold(precisions, recalls, thresholds, beta=0.5)

    # Apply threshold to probability predictions to adjust classification decisions
    adjusted_predictions = (probabilities >= optimal_threshold).astype(int)
    
    # Store scores
    acc_scores.append(accuracy_score(y_test, adjusted_predictions))
    recall_scores.append(recall_score(y_test, adjusted_predictions, average='macro'))
    precision_scores.append(precision_score(y_test, adjusted_predictions, average='macro'))
    f1_scores.append(f1_score(y_test, adjusted_predictions, average='macro'))

print("Model with SMOTE, KFOLD and ADJUSTED THRESHOLD")
# Calculate mean of each metric across all folds
print('Adjusted Accuracy: %.3f' % mean(acc_scores))
print('Adjusted Precision: %.3f' % mean(precision_scores))
print('Adjusted Recall: %.3f' % mean(recall_scores))
print('Adjusted F1 Score: %.3f' % mean(f1_scores))

print('--------------------------------------------------------------------------------')
print('--------------------------------------------------------------------------------')

# Define the final model using the pipeline configuration tested
final_model = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=58)),
    ('classifier', GaussianNB())
])

# Train the model on the entire dataset
final_model.fit(X_smote, y_smote)

import pickle

# Serialize the model to a file for deployment
with open('naivebayes_trained_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)












