# %% [markdown]
# # Group Project - KSI data - Classification problem

# %% [markdown]
# ***Target Column***
# ACCLASS<br>
# Required to transform into binary (0, 1):<br>
# 'Fatal' --> 1, <br>
# 'Non-Fatal Injury' --> 0, <br>
# 'Property Damage Only' --> 0, <br>
# ***5 nan value from this column, we can consider to drop them***
# 
# 
# 
# below columns need to fill values:
# 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
# 'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV',
# 'REDLIGHT', 'ALCOHOL', 'DISABILITY'
# fill Nan as No, and transform to 0, 1
# (Default they are Yes, Nan values)
# ROAD_CLASS fill most freq value
# DISTRICT fill most freq value
# 
# Questionable column:
# CYCCOND: multi categories, fill Nan as most freq value??
# 
# 
# From the dataset, below columns are unnecessary:
# ObjectId, HEIGHBOURHOOD_158, HEIGHBOURHOOD_140, CYCLISTYPE(too much categories and too much Nan value),<br>
# PEDCOND(too much categories and too much Nan value), PEDACT(too much categories and too much Nan value),<br>
# PEDTYPE (too much categories and too much Nan value), DRICOND ('other' included, means it is not a accuracy value), DRIVACT ('other' included, means it is not a accuracy value), MANOEUVER('other' included, means it is not a accuracy value)<br>
# FATAL_NO, INVTYPE, DATE, YEAR, ACCNUM, INDEX_, STREET1, STREET2, OFFSET, X, Y,INJURY
# 
# 

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
dataset_path = r'dataset\KSI.csv'

df = pd.read_csv(dataset_path)

# %%
df.head(5)

# %%
# Drop rows which YEAR is 2020 or 2021 (COVID-19 pandemic)
#df = df[~df['YEAR'].isin([2020, 2021])]

# %%
# Drop rows which has duplicate ACCNUM, same ACCNUM means same accident
#df = df.drop_duplicates(subset='ACCNUM', keep='first')

# %% [markdown]
# # Exploration <br>
# Use below code to display categrial data and null counts

# %%
print(df['ROAD_CLASS'].value_counts())
print(df['ROAD_CLASS'].isnull().sum())

# %%
df.info()

# %%
df.columns.values

# %% [markdown]
# # Determine columns

# %%
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



# %%
# Create a copy of the dataframe
df_origin = df.copy()

# %%
#drop meaningless columns, duplicated columns and columns with too much missing values
df = df.drop(columns=meaningless_columns)
df = df.drop(columns=duplicated_columns)
df = df.drop(columns=too_much_missing)


# %% [markdown]
# # Dealing with columns which contains many catefories

# %%
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



# %%
print(df.columns.values)
print(len(df.columns.values))


# %%
# columns can try to exclude or include

#selected_categorical_columns = set(categorical_columns) - set(try_exclude)
selected_categorical_columns = categorical_columns
print(selected_categorical_columns)

# %%
# join selected_categorical_columns and fill_nan_columns
selected_cat_columns = list(selected_categorical_columns) + fill_nan_columns

print(selected_cat_columns)

# %%
#df = df.drop(columns=try_exclude, axis=1)

# %%
df[fill_nan_columns] = df[fill_nan_columns].fillna(value='No')

# %%
df.info()

# %%
#export the cleaned dataset
df.to_csv(r'dataset\KSI_cleaned.csv', index=False)

# %%
X = df.drop(columns=['ACCLASS'], axis=1)
y = df['ACCLASS']

data_map = {
    'Fatal': 1, 
    'Non-Fatal Injury': 0, 
    'Property Damage Only': 0
    }

y = y.map(data_map)


# %%
type(X), type(y)

# %%
df.dtypes

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE




# %%
#Preprocessing for categorical data
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    
])

#Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', cat_transformer, selected_cat_columns),
    ])

#get transformed data
X_transformed = preprocessor.fit_transform(X)
X_transformed = pd.DataFrame(X_transformed)
print(X_transformed.shape, y.shape)

#Resample the data
smote = SMOTE(random_state=58)
X_resampled, y_resampled = smote.fit_resample(X_transformed, y)


# %%
y_resampled.value_counts()

# %%
#split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=58)

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


#Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[
                      ('model', model)
                     ])

#Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# %%
#Preprocessing of validation data, get predictions
y_preds = clf.predict(X_test)

#Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy:', accuracy_score(y_test, y_preds))

print(confusion_matrix(y_test, y_preds))

print(classification_report(y_test, y_preds))

# %% [markdown]
# # Testing with different classifier

# %%
# Use SearchCV to find best estimator with Logistic Regression model

from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

y_preds = grid_search.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_preds))




# %%
# Create SVM model, and use GridSearchCV to find best estimator

from sklearn.svm import SVC

model = SVC()

clf = Pipeline(steps=[
        ('model', model)
    ])

param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

y_preds = grid_search.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_preds))



# %% [markdown]
# # Create Random Forest model, and use GridSearchCV to find best estimator
# 
# from sklearn.ensemble import RandomForestClassifier
# 
# model = RandomForestClassifier()
# 
# clf = Pipeline(steps=[
#         ('model', model)
#     ])
# 
# param_grid = {
# 
#     'model__n_estimators': [100, 200, 300, 400, 500],
#     'model__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'model__min_samples_split': [2, 5, 10],
#     'model__min_samples_leaf': [1, 2, 4],
#     'model__bootstrap': [True, False]
# }
# 
# grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=2)
# 
# grid_search.fit(X_train, y_train)
# 
# print(grid_search.best_params_)
# 
# y_preds = grid_search.predict(X_test)
# 
# print('Accuracy:', accuracy_score(y_test, y_preds))
# 
# 

# %% [markdown]
# 


